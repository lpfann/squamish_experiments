def get_truth(params):
    strong = params["strong"]
    weak = params["weak"]
    irrel = params["irr"]
    truth = [True] * (strong + weak) + [False] * irrel
    return truth

