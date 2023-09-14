import logging


def log_sequence(seq, indent="") -> None:
    for element in seq:
        logging.info(f"{indent}{element}")

def count_parameters(model):
    """
    Based on https://stackoverflow.com/questions/49201236/check-the-total-number-of-parameters-in-a-pytorch-model,
    forwarded to me by Jonas Groschwitz
    """
    from prettytable import PrettyTable
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        params = parameter.numel()
        table.add_row([name, params])
        total_params += params
    print(table)
    print(f"Total Trainable Params: {total_params}")
    return total_params
