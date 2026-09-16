//! Dedicated library for simulating control systems.
//!
//! Control systems are written as continuous-time differential equations that
//! receive input signals that (by default) are zeroth-order hold through each
//! timestep.

/// Convenience export of an integrator for use in control system simulations.
use high_gain_observer::{Float, Time, Timestampable, Vectorizable};

/// Exposed integrator for this library.
pub struct Integrator(bulirsch::AdaptiveIntegrator<Float>);

impl Default for Integrator {
    /// A standard numerical integrator suitable for most uses.
    fn default() -> Self {
        Self(
            bulirsch::Integrator::default()
                .with_abs_tol(1e-8)
                .with_rel_tol(1e-8)
                .into_adaptive(),
        )
    }
}

/// System: implements "step" method which integrates and produces a new state
/// and output.
pub trait ControlSystem {
    type State: Vectorizable;
    type Params;
    type Output: Timestampable;
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
        let mut final_state = ndarray::Array::zeros([Self::State::size()]);
        // Define a ZOH system for the provided input.
        let system = ZOHSystem::<Self> { input, params };
        // Integrate it for the time window.
        let _stats = integrator.0.step(
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

impl<C: ControlSystem> bulirsch::System for ZOHSystem<'_, C> {
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
