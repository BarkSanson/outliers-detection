import tabulate
import os


class Printer:
    def __init__(self, results_path):
        self._scores_path = os.path.join(results_path, "scores")
        os.makedirs(self._scores_path, exist_ok=True)

    def print_scores(self, models, results, window_size):
        # Create a table with the accuracy of each model
        for model, res in results.items():
            current_model = filter(lambda x: x.name == model, models)
            model_params = next(current_model).params

            params_names = list(model_params.keys())

            # Map the results to a list of lists, where each list is a row in the table
            data = list(map(lambda x: [*x[1:-1], x[-1]], res))

            with open(os.path.join(self._scores_path, f'{model}_with_window_size_{window_size}.txt'), 'w') as f:
                f.write(tabulate.tabulate(
                    data,
                    headers=[*params_names, 'Precision', 'Recall', 'F1 score'],
                    tablefmt='orgtbl'))

