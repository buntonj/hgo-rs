//! Usage of a high-gain observer in Rust.

use high_gain_observer::{
    Float, HighGainObserver, HighGainObserverParams, Observer, Time, Timestampable, Vectorizable,
};

/// Integrator for the example.
type Integrator = bulirsch::AdaptiveIntegrator<Float>;

/// System: implements "step" method which integrates and produces a new state
/// and output.
pub trait ControlSystem {
    type State: Vectorizable;
    type Params;
    type Output;
    type Input;
    /// Step method.  Calling this will produce a new state and output assuming
    /// the input is held fixed for the entire dt.
    ///
    /// The default implementation will perform a zero-order-hold of the input
    /// for the time window.
    fn step(
        &self,
        state: &Self::State,
        params: &Self::Params,
        input: &Self::Input,
        integrator: &mut Integrator,
        start_time: Time,
        end_time: Time,
    ) -> (Self::State, Self::Output)
    where
        Self: Sized,
    {
        let timestep_sec = (end_time - start_time).to_seconds();
        tracing::info!("Stepping truth system {timestep_sec} s");
        let mut final_state = ndarray::Array::zeros([Self::State::size()]);
        // Define a ZOH system for the provided input.
        let system = ZOHSystem::<Self> { input, params };
        // Integrate it for the time window.
        let _stats = integrator.step(
            &system,
            timestep_sec,
            state.vectorize().view(),
            final_state.view_mut(),
        );
        let packed_state = Self::State::devectorize(final_state.view());
        let output = self.compute_output(&packed_state, input, params);
        (packed_state, output)
    }

    fn time_derivative(
        state: &Self::State,
        input: &Self::Input,
        params: &Self::Params,
    ) -> Self::State;

    fn compute_output(
        &self,
        state: &Self::State,
        input: &Self::Input,
        params: &Self::Params,
    ) -> Self::Output;
}

/// A struct we build with a fn pointer for the RHS frozen with the input
struct ZOHSystem<'a, C: ControlSystem> {
    /// The input to use for the time window.
    input: &'a C::Input,
    /// The params for the system.
    params: &'a C::Params,
}

impl<'a, C: ControlSystem> bulirsch::System for ZOHSystem<'a, C> {
    type Float = Float;
    fn system(
        &self,
        y: ndarray::ArrayView1<Self::Float>,
        mut dydt: ndarray::ArrayViewMut1<Self::Float>,
    ) {
        dydt.assign(
            &(C::time_derivative(&C::State::devectorize(y), self.input, self.params).vectorize()),
        );
    }
}

/// System representing a spinning wheel that has torque inputs, e.g., a
/// reaction wheel.
struct TorqueInputSystem;

/// Parameters of the system.
#[derive(Debug)]
struct TorqueSystemParams {
    /// Inertia of the rotating system.
    inertia_kg_m2: Float,
    /// Measurement noise stddev.
    _measurement_noise_stddev: Float,
    /// RNG seed for the measurement noise.
    _rng: rand::rngs::SmallRng,
}

/// State of the torque-driven system.
#[derive(Clone, Debug, Default)]
struct TorqueSystemState {
    /// Position of the wheel in radians.
    position_rad: Float,
    /// Angular velocity of the wheel in radians.
    angular_velocity_rad_per_s: Float,
}

// TODO: This could be derive-macro'd away.
impl Vectorizable for TorqueSystemState {
    fn devectorize(vector: ndarray::ArrayView1<Float>) -> Self {
        TorqueSystemState {
            position_rad: vector[0],
            angular_velocity_rad_per_s: vector[1],
        }
    }
    fn vectorize(&self) -> ndarray::Array1<Float> {
        ndarray::Array::from_iter([self.position_rad, self.angular_velocity_rad_per_s])
    }
    fn size() -> usize {
        2
    }
}

impl ControlSystem for TorqueInputSystem {
    type Input = Float;
    type Output = Float;
    type State = TorqueSystemState;
    type Params = TorqueSystemParams;

    fn time_derivative(
        state: &Self::State,
        input: &Self::Input,
        params: &Self::Params,
    ) -> Self::State {
        TorqueSystemState {
            // Position derivative is angular velocity
            position_rad: state.angular_velocity_rad_per_s,
            // Velocity derivative is 1/J * torque
            angular_velocity_rad_per_s: input / params.inertia_kg_m2,
        }
    }

    fn compute_output(
        &self,
        state: &Self::State,
        _input: &Self::Input,
        _params: &Self::Params,
    ) -> Self::Output {
        state.position_rad
    }
}

/// Step method components to estimate value of state at time T:
/// 1. Measurement is timestamped at time t < T.
/// 2. Integrate state up until time t ("open loop") (if not already there).
/// 3. Compute the measurement error--estimated output vs real.
/// 4. Integrate the state with measurement error correction from t to T using forward euler.
///
/// Could consider a Tustin transform of the dynamics to integrate?  Need some continuous -> discrete method.
/// Khalil papers traditionally just use forward euler.
/// Could make agnostic/selectable?
/// This is purely internal--we are provided with dynamics that are used to build observer dynamics, and can integrate them however tf we want.
/// Start simple--forward euler is easy and foolproof

/// Now we can simulate a control system with arbitrary parameters and steps!
/// May be nice if we can put the integrator/parameters into the system itself.
///
/// Next step: define an observer that can consume those measurements!
fn main() {
    tracing_subscriber::fmt::init();
    let system = TorqueInputSystem;
    let params = TorqueSystemParams {
        inertia_kg_m2: 1.0,
        _measurement_noise_stddev: 0.1,
        _rng: <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(666),
    };
    let state = TorqueSystemState {
        position_rad: 0.0,
        angular_velocity_rad_per_s: 0.0,
    };

    let torque_input = 1.0;

    let mut integrator = bulirsch::Integrator::default()
        .with_abs_tol(1e-8)
        .with_rel_tol(1e-8)
        .into_adaptive();
    let start_time = hifitime::Epoch::from_tai_seconds(0.0);
    let mut current_time = start_time.clone();

    let mut observer = HighGainObserver::new(
        HighGainObserverParams {
            gains: ndarray::Array1::from_vec(vec![2.5, 1.0]),
            epsilon: 0.1,
        },
        TorqueSystemState::default(),
        current_time,
    );

    for _ in 0..100 {
        let end_time = current_time + hifitime::Duration::from_seconds(0.1);
        let (state, output) = system.step(
            &state,
            &params,
            &torque_input,
            &mut integrator,
            current_time,
            end_time,
        );
        // Step the observer.
        observer.step(output.timestamp(end_time), end_time);
        let current_estimate = observer.current_estimate();
        let current_estimate_state = current_estimate.as_state();
        current_time = end_time;
        let elapsed_s = (current_time - start_time).to_seconds();
        tracing::info!("T = {elapsed_s}:");
        tracing::info!("Truth: {state:?}");
        tracing::info!("Estimate: {current_estimate_state:?}");
    }
}
