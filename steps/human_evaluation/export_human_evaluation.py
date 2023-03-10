import warnings

from steps.human_evaluation import export_samples


__EXECUTION_DICT = {
    1: export_samples.export_clusters_sample_images()
    2: 
}

__MENU = """
exit: terminate the program
0: Execute all

1: Export the MNIST human evaluation
2: Export the IMDB human evaluation

Select one or more of the options separated by a space: """

__INVALID_OPTION = lambda option: f'Invalid option [{option}]'

if __name__ == '__main__':
    # Add the choice for execute all
    __EXECUTION_DICT[0] = __EXECUTION_DICT.values()
    warnings.filterwarnings('ignore')
    choices_str = input(__MENU)
    while choices_str != 'exit':
        choices = []
        try:
            choices = [int(choice) for choice in choices_str.split(' ')]
        except ValueError:
            print(__INVALID_OPTION(choices_str))

        # Get the handler for the input
        for choice in choices:
            if choice not in __EXECUTION_DICT.keys():
                print(__INVALID_OPTION(choice))
            else:
                handler = __EXECUTION_DICT.get(choice)
                if handler is not None:
                    try:
                        handler()
                    except TypeError:
                        [handler_item() for handler_item in list(handler)[:-1]]

        choices_str = input(__MENU)
