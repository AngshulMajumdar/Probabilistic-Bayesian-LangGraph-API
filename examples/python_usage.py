from b_langgraph.scenarios.registry import build_stale_vs_verified


def main() -> None:
    agent = build_stale_vs_verified()
    answer, trace, posterior = agent.run(
        'What is the best time to visit Serbia? Consider weather and budget.'
    )
    print('Answer:', answer)
    print('Steps:', len(trace.steps))
    print('Particles:', posterior['n_particles'])


if __name__ == '__main__':
    main()
