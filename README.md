# High-gain observers in Rust
Casual exploration of the implementation of high-gain observers in Rust. **Buyer beware.**

High-gain observers are a parameterized family of state estimators for a fairly general class of nonlinear systems. For now, I've lazily only handled systems of the form:

$$ \dot{x}_i = x_{i+1}, \\
\dot{x}_n = \phi(x,u(t)), \\
y(t) = x_1. $$

Here $x \in\mathbb{R}^n$ represents the system "state", $u(t) \in\mathbb{R}^m$ represents a control input, $\phi: \mathbb{R}^n \times \mathbb{R}^m \to\mathbb{R}$ is some nonlinear relationship between the control input and the dynamics, and $y(t)\in\mathbb{R}$ is an output.  High-gain observers can be modified to handle far more general classes of systems than this, but one thing at a time!

A basic high-gain observer for the above system takes the form:

$$ \dot{\hat{x}}_i = x_{i+1} - \frac{\alpha_i}{\varepsilon^i}(y-x_1), \quad i = 1, 2,...,n-1 \\
\dot{\hat{x}}_n = \hat{\phi}(\hat{x}, u) + \frac{\alpha_n}{\varepsilon^n}(y-x_1)$$

Here $\hat{x}\in\mathbb{R}^n$ is our state _estimate_, and $\hat{\phi}$ is some "nominal" nonlinearity model that we leverage. Both $\alpha\in\mathbb{R}^n$ and $\varepsilon\in\mathbb{R}_{>0}$ are parameters to tweak for observer convergence--typically one chooses $\alpha$ such that the roots of the characteristic polynomial:

$$ s^n + \alpha_1 s^{n-1} + \alpha_2 s^{n-2} + ... + \alpha_{n-1}s + \alpha_n$$

have strictly negative real part.  The parameter $\varepsilon > 0$ is the "high gain" part of the high-gain observer, which should be set as small as possible without causing explosive results (note that it enters the observer dynamics with powers up to $\varepsilon^{-n}$).

For simplicity, I elected to let $\phi(x, u) = 0$ in this implementation, but of course, including a more representative input/nonlinearity model gives better performance.  The intuition is that for sufficiently high gain (read: mix of short timescale and forced quick convergence) a mostly linear and "chain of integrators" observer can "ignore" any of the nonlinear effects of dynamics.

## What's still left

Some things to play with:
* Split off the utilities I added for just "general control systems integrators" (`ControlSystem` trait, for exammple) so examples can be more lean.
* Handle more general dynamics than a chain of integrators for the provided observer:
    * There can be a $\psi_i(x_1,...,x_i)$ additive nonlinearity that must be known and shared with the observer.
    * Multi-output is just "make N copies of the observer" IIRC, so we could just auto-do that.
* Consider a more general integration scheme than Euler for the observer itself (or just support more general if desired).
* The Dr. Pauline Bernard has a neat alternate form of these observers that live directly in discrete time (rather than requiring some discretization scheme).
* Various references have added tricks to reduce the gain of these observers:
    * Khalil's book has a simple trick to reduce dimensionality by one.
    * Astolfi has a paper on reducing the gain powers by half with some sneaky state duplication.
* Add better visualization tools (just looking at terminal tracing outputs _sucks_)
* More examples!!
* Add some utilities to help design the observer:
    * Simple pole placement with all eigenvalues at the same number (due to structure, this is easy and doesn't require general pole placement algorithms, or we could do Lyapunov/
    * Heuristics for cranking the gain $\varepsilon$ the right amount (or some warning about error)