import json
import math
from argparse import ArgumentParser
from pathlib import Path

from lpinstance import LPSolver, diet_problem
from model_timer import Timer


# Stencil created by Anirudh Narsipur March 2023


def main(args):
    # diet_problem()

    filename = Path(args.input_file).name
    timer = Timer()
    timer.start()
    lp_solver = LPSolver(args.input_file)
    sol = lp_solver.solve()
    timer.stop()

    printSol = {
        "Instance": filename,
        "Time": timer.get_elapsed(),
        "Result": math.ceil(sol),
        "Solution": "OPT"
    }

    print(json.dumps(printSol))


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("input_file", type=str)
    arguments = parser.parse_args()
    main(arguments)
