//! Implementation of a high-gain observer in Rust.

/// Timestamp type for the project.
pub type Time = hifitime::Epoch;

/// Floating point type for this crate.
pub type Float = f64;

/// Integrator for the project.
pub type Integrator = bulirsch::AdaptiveIntegrator<Float>;

/// Trait for converting from a structured state/input/etc into a 1D array.
pub trait Vectorizable {
    fn size() -> usize;
    fn vectorize(&self) -> ndarray::Array1<Float>;
    fn devectorize(vector: ndarray::ArrayView1<Float>) -> Self;
}

/// System: implements "step" method which integrates and produces a new state
/// and output.
pub trait ControlSystem {
    type State: Vectorizable;
    type Params;
    type Output;
    type Input;
    /// Step method.  Calling this will produce a new state and output assuming the input is held fixed for the entire dt.
    fn step(
        &self,
        state: &Self::State,
        params: Self::Params,
        input: Self::Input,
        mut integrator: Integrator,
        start_time: Time,
        end_time: Time,
    ) -> (Self::State, Self::Output)
    where
        Self: Sized,
    {
        let mut final_state = ndarray::Array::zeros([Self::State::size()]);
        // Define a ZOH system for the provided input
        let system: ZOHSystem<Self> = ZOHSystem { params, input };
        // Integrate it
        let _stats = integrator.step(
            &system,
            (end_time - start_time).to_seconds(),
            state.vectorize().view(),
            final_state.view_mut(),
        );
        let packed_state = Self::State::devectorize(final_state.view());
        let output = Self::compute_output(&packed_state, &system.input, &system.params);
        (packed_state, output)
    }

    fn time_derivative(
        state: &Self::State,
        input: &Self::Input,
        params: &Self::Params,
    ) -> Self::State;

    fn compute_output(
        state: &Self::State,
        input: &Self::Input,
        params: &Self::Params,
    ) -> Self::Output;
}

/// Concrete struct for a control system operating in "zero-order-hold" mode.
///
/// Wraps a given `ControlSystem`-implementing system.
struct ZOHSystem<C: ControlSystem> {
    /// Current input to apply over time window.
    input: C::Input,
    params: C::Params,
}

impl<C: ControlSystem> bulirsch::System for ZOHSystem<C> {
    type Float = Float;
    fn system(
        &self,
        y: bulirsch::ArrayView1<Self::Float>,
        mut dydt: bulirsch::ArrayViewMut1<Self::Float>,
    ) {
        dydt.assign(
            &(C::time_derivative(&C::State::devectorize(y), &self.input, &self.params).vectorize()),
        );
    }
}

/// System representing a spinning wheel that has torque inputs, e.g., a
/// reaction wheel.
struct TorqueInputSystem;

/// Parameters of the system.
#[derive(Debug)]
struct TorqueSystemParams {
    inertia_kg_m2: Float,
}

/// State of the torque-driven system.
#[derive(Debug)]
struct TorqueSystemState {
    /// Position of the wheel in radians.
    position_rad: Float,
    /// Angular velocity of the wheel in radians.
    angular_velocity_rad_per_s: Float,
}

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
        state: &Self::State,
        _input: &Self::Input,
        _params: &Self::Params,
    ) -> Self::Output {
        state.position_rad
    }
}

fn main() {
    let system = TorqueInputSystem;
    let params = TorqueSystemParams { inertia_kg_m2: 1.0 };
    let state = TorqueSystemState {
        position_rad: 0.0,
        angular_velocity_rad_per_s: 0.0,
    };

    let torque_input = 1.0;

    let integrator = bulirsch::Integrator::default()
        .with_abs_tol(1e-8)
        .with_rel_tol(1e-8)
        .into_adaptive();
    let start_time = hifitime::Epoch::from_tai_seconds(0.0);
    let end_time = start_time + hifitime::Duration::from_seconds(60.0);

    let (final_state, output) = system.step(
        &state,
        params,
        torque_input,
        integrator,
        start_time,
        end_time,
    );
    println!("Stepped from {start_time} to {end_time}");
    println!("Initial state: {state:?}");
    println!("Final state:   {final_state:?}");
    println!("Output:        {output:?}")
}
