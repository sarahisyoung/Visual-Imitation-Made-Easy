
class EarlyStop:

    def __init__(self, patience=10, delta=0):
        self.patience = patience
        self.counter = 0
        self.min_loss = 1e9
        self.delta = delta

    def __call__(self, val_loss, model, epoch):
        """
            Returns boolean tuple (stop, save)
        """
        # Validation loss doesn't decrease.
        if val_loss >= self.min_loss + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                return True, False

        # Validation loss decreases.
        else:
            print("Validation loss decreased at epoch " + str(epoch) + ": (" + '{:.6}'.format(
                self.min_loss) + " --> " + '{:.6}'.format(val_loss) + ").")
            self.min_loss = val_loss
            self.counter = 0
            return False, True
        
        return False, False

