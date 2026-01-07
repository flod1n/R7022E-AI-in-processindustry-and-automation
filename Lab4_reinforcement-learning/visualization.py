import pcgym
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import rcParams


def get_data_from(policies: dict, env: pcgym.make_env, data_file_name: str) -> None:
    evaluator, data = env.get_rollouts(policies, reps=50, oracle=True, MPC_params={'N':17})
    data['reference'] = {}
    for k, sp in env.SP.items():
        data['reference'][k] = np.asarray(sp)
    np.save(data_file_name, data)


def performance_plots(data, policies):
    # Set up LaTeX rendering
    rcParams['text.usetex'] = True
    rcParams['font.family'] = 'serif'
    rcParams['axes.labelsize'] = 10
    rcParams['xtick.labelsize'] = 10
    rcParams['ytick.labelsize'] = 10
    rcParams['legend.fontsize'] = 10

    t = np.linspace(0, 25, 60)

    # A4 width in inches
    a4_width_inches = 8.27

    # Calculate height to maintain aspect ratio
    height = a4_width_inches * 0.4  # Adjust this factor as needed

    fig, axs = plt.subplots(1, 3, figsize=(a4_width_inches, height))
    plt.subplots_adjust(wspace=0.3, top=0.85, bottom=0.15, left=0.08, right=0.98)

    policies.insert(0, 'oracle')
    cols = ['tab:orange', 'tab:red', 'tab:blue', 'tab:green']
    # labels = ['Oracle', 'SAC', 'PPO', 'DDPG']

    # Create lines for the legend
    lines = []
    for i, policy in enumerate(policies):
        line, = axs[0].plot([], [], color=cols[i], label=policies[i])
        lines.append(line)
    ref_line, = axs[0].plot([], [], color='black', linestyle='--',  label='Reference')
    lines.append(ref_line)

    # Create legend above the plots
    fig.legend(handles=lines, loc='upper center', 
                bbox_to_anchor=(0.37, 0.94),
                ncol=5, 
                frameon=False, 
                columnspacing=1)

    y_labels = [r'$C_A$ [mol/m$^3$]', r'$T_c$ [K]']

    for idx, ax in enumerate(axs):
        ax.grid(True, linestyle='--', alpha=0.7)
        ax.set_axisbelow(True)

        # --- STATE ---
        if idx == 0:
            for i, policy in enumerate(policies):
                ax.plot(
                    t,
                    np.median(data[policy]['x'][0,:,:], axis=1),
                    color=cols[i],
                    linewidth=1.25
                )
                ref = data['reference']['Ca'][:len(t)]
                ax.plot(
                    t,
                    ref,
                    color='black',
                    linestyle='--',
                    linewidth=1.5
                )
                ax.fill_between(
                    t,
                    np.max(data[policy]['x'][0,:,:], axis=1),
                    np.min(data[policy]['x'][0,:,:], axis=1),
                    alpha=0.2,
                    color=cols[i]
                )
            ax.set_ylabel(r'$C_A$ [mol/m$^3$]')
            ax.set_xlabel(r'Time [min]')
            ax.set_xlim(0, 25)

        if idx == 1:
            for i, policy in enumerate(policies):
                ax.step(t, np.median(data[policy]['u'][0,:,:], axis=1), color=cols[i], where='post',linewidth=1.25)
                ax.fill_between(t, np.max(data[policy]['u'][0,:,:], axis=1),
                                np.min(data[policy]['u'][0,:,:], axis=1),
                                step="post", alpha=0.2, linewidth=0, color=cols[i])
                ax.set_ylabel(y_labels[idx])
                ax.set_xlabel(r'Time [min]')
                ax.set_xlim(0, 25)
        if idx == 2:
            # Calculate oracle reward
            oracle_reward = np.median(data['oracle']["r"].sum(axis=1))

            # Calculate normalized optimality gap for each policy (for statistics only)
            normalized_gaps = {}
            for policy in policies[1:]:  # Exclude oracle
                gaps = (oracle_reward - data[policy]["r"].sum(axis=1))
                normalized_gaps[policy] = gaps.flatten()

            # Calculate median absolute deviation and median normalized optimality gap
            mad = {}
            for policy in policies[1:]:
                mad[policy] = np.median(np.abs(data[policy]['r'].sum(axis=1) - np.median(data[policy]['r'].sum(axis=1))))
                print(f"{policy}:")
                print(f"  Median Absolute Deviation (MAD): {mad[policy]:.4f}")

            # Plot histograms using non-normalized reward values
            all_rewards = np.concatenate([data[policy]["r"].sum(axis=1).flatten() for policy in policies])
            min_reward, max_reward = np.min(all_rewards), np.max(all_rewards)
            bins = np.linspace(min_reward, max_reward, 21)

            for i, policy in enumerate(policies[1:], start=1):
                ax.hist(
                    data[policy]["r"].sum(axis=1).flatten(),
                    bins=bins,
                    color=cols[i],
                    alpha=0.5,
                    label=policies[i],
                    edgecolor='None',
                )
            ax.axvline(x=oracle_reward, color=cols[0], linestyle='--', linewidth=2, label='Oracle')
            ax.set_ylabel('Frequency')
            ax.set_xlabel('Cumulative Reward')

            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2, frameon=False)

    # Adjust the plots to be square and the same size
    for ax in axs:
        ax.set_box_aspect(1)

    plt.savefig('cstr_vis.pdf', bbox_inches='tight', pad_inches=0.1)
    plt.show()