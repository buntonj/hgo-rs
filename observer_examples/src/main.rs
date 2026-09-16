//! Usage of a high-gain observer in Rust.

use control_system::ControlSystem;
use high_gain_observer::{
    Float, HighGainObserver, HighGainObserverParams, Observer, Timestampable, Vectorizable,
};

/// Directory for output artifacts to get dumped.
const ARTIFACT_DIR: &str = "./artifacts";

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

/// Simulate a system with torque inputs and a high-gain observer consuming those measurements.
fn main() {
    tracing_subscriber::fmt::init();
    let system = TorqueInputSystem;
    let params = TorqueSystemParams {
        inertia_kg_m2: 1.0,
        _measurement_noise_stddev: 0.1,
        _rng: <rand::rngs::SmallRng as rand::SeedableRng>::seed_from_u64(666),
    };
    let mut state = TorqueSystemState {
        position_rad: 0.0,
        angular_velocity_rad_per_s: 0.0,
    };
    let mut output: <TorqueInputSystem as ControlSystem>::Output;

    let torque_input = 1.0;

    let mut integrator = control_system::Integrator::default();
    let start_time = hifitime::Epoch::from_tai_seconds(0.0);
    let mut current_time = start_time;

    let mut observer = HighGainObserver::new(
        HighGainObserverParams {
            gains: ndarray::Array1::from_vec(vec![2.0, 1.0]),
            epsilon: 1.0,
        },
        &TorqueSystemState::default(),
        start_time,
    );

    let mut times = Vec::new();
    let mut truth_states = Vec::new();
    let mut estimate_states = Vec::new();

    for _ in 0..100 {
        let end_time = current_time + hifitime::Duration::from_seconds(0.1);
        (state, output) = system.step(
            &state,
            &params,
            &torque_input,
            &mut integrator,
            current_time,
            end_time,
        );
        let measurement_time = end_time;
        // Step the observer.
        observer.step_to(output.timestamp(measurement_time), torque_input, end_time);
        let current_estimate = observer.current_estimate();
        let current_estimate_state = current_estimate.as_state();
        current_time = end_time;
        let elapsed_s = (current_time - start_time).to_seconds();
        tracing::info!("T = {elapsed_s}s:");
        tracing::info!("Truth: {state:?}");
        tracing::info!("Estimate: {current_estimate_state:?}");

        times.push((current_time - start_time).to_seconds());
        truth_states.push(state.clone());
        estimate_states.push(current_estimate_state.clone());

        let error = TorqueSystemState::devectorize(
            (state.vectorize() - current_estimate_state.vectorize()).view(),
        );
        tracing::info!("Error: {error:?}");
    }

    let mut plot = plotly::Plot::new();
    plot.add_trace(
        plotly::Scatter::new(
            times.clone(),
            truth_states
                .iter()
                .zip(estimate_states)
                .map(|(truth, estimate)| truth.position_rad - estimate.position_rad)
                .collect(),
        )
        .name("Position estimate error"),
    );
    std::fs::create_dir_all(ARTIFACT_DIR).unwrap();
    plot.write_html(format!("{ARTIFACT_DIR}/out.html"));
}
