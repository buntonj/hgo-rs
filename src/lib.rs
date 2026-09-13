//! Implementation of a high-gain observer in Rust.

/// Floating point type for this crate.
pub type Float = f64;

/// Timestamp type for the project.
pub type Time = hifitime::Epoch;

/// Timestamped element (for convenience).
pub struct Timestamped<T> {
    /// Value.
    pub value: T,
    /// Timestamp associated with this value.
    pub time: Time,
}

/// Convenience trait for timestamping values.
///
/// Blanket impl'd for all Sized types for easy use.
pub trait Timestampable {
    fn timestamp(self, time: Time) -> Timestamped<Self>
    where
        Self: Sized,
    {
        Timestamped { value: self, time }
    }
}

impl<T: Sized> Timestampable for T {}

/// Trait for converting from a structured state/input/etc into a 1D array.
pub trait Vectorizable {
    fn size() -> usize;
    fn vectorize(&self) -> ndarray::Array1<Float>;
    fn devectorize(vector: ndarray::ArrayView1<'_, Float>) -> Self;
}

/// High-gain observer for a given control system.
///
/// Requires some parameters to instantiate.
/// Consumes measurements, inputs, and a timestep size with each call to "step", returning an estimate.
///
/// High gain observers only apply to systems that are uniformly observable.
/// We'll need to add a trait/method for systems that are diffeomorphic to a
/// chain of integrators, or assume the system is already in this form.  Likely
/// the latter, leaving the implementation details of that diffeomorphism to the
/// user.  Easy enough to implement for an example though.
///
///
pub trait Observer {
    type Input;
    type Estimate;
    type Measurement;

    /// Step the observer.
    ///
    /// Given a measurement with the provided timestep, produce a state estimate
    /// at the requested time.
    fn step_to(
        &mut self,
        measurement: Timestamped<Self::Measurement>,
        input: Self::Input,
        time: hifitime::Epoch,
    );

    /// Extract the current estimate type.
    fn current_estimate(&self) -> Self::Estimate;
}

type HighGainObserverMeasurement = Float;

#[derive(Clone)]
struct HighGainObserverState<T: Vectorizable + Clone> {
    /// The state vector (transformed), q in the Khalil reference.
    vector: ndarray::Array1<Float>,
    /// The last received measurement.
    last_measurement: Option<HighGainObserverMeasurement>,
    /// The current input (currently only single input).
    input: Option<Float>,
    /// The current time.
    time: hifitime::Epoch,
    /// Phantom data representing the original state type.
    phantom: core::marker::PhantomData<T>,
}

impl<T: Vectorizable + Clone> HighGainObserverState<T> {
    /// Create a new `HighGainObserverState` from the given `Vectorizable` state.
    pub fn new(state: T, time: hifitime::Epoch) -> Self {
        Self {
            vector: state.vectorize(),
            last_measurement: None,
            input: None,
            time,
            phantom: core::marker::PhantomData,
        }
    }
}

#[derive(Debug)]
pub struct HighGainObserverEstimate<T: Vectorizable> {
    /// The state vector (in natural coordinates).
    vector: ndarray::Array1<Float>,
    /// The time this estimate is associated with.
    time: hifitime::Epoch,
    /// Phantom data representing what state struct this estimate corresponds to.
    phantom: core::marker::PhantomData<T>,
}

impl<T: Vectorizable> HighGainObserverEstimate<T> {
    /// Convert the current estimate into the state struct.
    pub fn as_state(&self) -> T {
        T::devectorize(self.vector.view())
    }

    /// Return the state struct and timestamp for the estimate.
    pub fn into_timestamped_state(&self) -> Timestamped<T> {
        let time = self.time;
        self.as_state().timestamp(time)
    }
}

/// Parameters of the observer.
pub struct HighGainObserverParams {
    /// Array of gains (h_i in the Khalil reference).
    pub gains: ndarray::Array1<Float>,
    /// The "sufficiently small" value to crank gains.
    pub epsilon: Float,
}

/// The observer.
pub struct HighGainObserver<T: Vectorizable + Clone> {
    /// Current state.
    state: HighGainObserverState<T>,
    /// Parameters of the observer.
    params: HighGainObserverParams,
}

impl<T: Vectorizable + Clone> HighGainObserver<T> {
    /// Create a new observer with the given parameters and initial state.
    pub fn new(params: HighGainObserverParams, state: T, time: Time) -> Self {
        Self {
            state: HighGainObserverState::new(state, time),
            params: params,
        }
    }

    // TODO: Add some form of `with_poles_at` hook to do automatic pole placement.

    /// Propagate the state forward using forward-Euler, with no new measurement.
    pub fn propagate(&mut self, to_time: hifitime::Epoch) {
        let timestep_seconds = (to_time - self.state.time).to_seconds();
        // Only do math if we are stepping in time.
        if timestep_seconds.abs() > 0.0 {
            let mut new_state = self.state.clone();
            let est_error = self
                .state
                .last_measurement
                .map_or(0.0, |measurement| measurement - self.state.vector[0]);

            // Step the dynamics of the observer.
            for (i, element) in new_state.vector.indexed_iter_mut() {
                let gain = self.params.gains[i] / self.params.epsilon.powi((i + 1) as i32);
                // tracing::info!("Stepping state element {i} with gain {gain}, error {est_error}");
                // Euler integration of the state. Last element doesn't integrate any extra state.
                *element += timestep_seconds
                    * (self
                        .state
                        .vector
                        .get(i + 1)
                        .unwrap_or(&self.state.input.unwrap_or(0.0))
                        + gain * est_error);
            }
            new_state.time = to_time;

            self.state = new_state;
        }
    }
}

impl<T: Vectorizable + Clone> Observer for HighGainObserver<T> {
    type Measurement = HighGainObserverMeasurement;
    // Just single-input for now.
    type Input = Float;
    type Estimate = HighGainObserverEstimate<T>;

    fn step_to(
        &mut self,
        measurement: Timestamped<Self::Measurement>,
        input: Self::Input,
        time: hifitime::Epoch,
    ) {
        // Update the input state.
        self.state.input = Some(input);
        // Propagate up to the measurement time
        self.propagate(measurement.time);
        // Update the measurement state
        self.state.last_measurement = Some(measurement.value);
        // Propagate to the desired time (will include error correction)
        self.propagate(time);
    }

    /// Produce the estimate at the current time.
    fn current_estimate(&self) -> HighGainObserverEstimate<T> {
        HighGainObserverEstimate {
            vector: self.state.vector.clone(),
            time: self.state.time,
            phantom: core::marker::PhantomData,
        }
    }
}
