import numpy as np

def plot_stability(fig, ax, delta, omegas, bc_init, bc_opt):
    # Compute p(ω) across full range
    omega_curve = np.linspace(0, np.max(omegas)*1.05, 500)
    mask = delta * omega_curve <= 1
    omega_curve = omega_curve[mask]
    p_curve = (-1 + np.sqrt(1 - (delta * omega_curve)**2)) / delta

    # Plot boundary
    ax.plot(omega_curve, p_curve, 'k--', linewidth=1.0, label=r'Boundary $p(\omega)$')

    # Scatter b_c values (initial and optimized)
    ax.scatter(omegas, bc_init, label='Initial', alpha=0.6, s=2)
    ax.scatter(omegas, bc_opt, label='Optimized', alpha=0.6, s=2)

    ax.set_xlabel(r'$\omega$ [rad/s]')
    ax.set_ylabel(r"$b_c=p(\omega)-b'$")
    ax.legend(frameon=False)
    ax.grid(ls=':', lw=0.5)
    return fig, ax