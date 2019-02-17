import time

def time_since(start, end):
    hours, rem = divmod(end - start, 3600)
    minutes, seconds = divmod(rem, 60)
    return hours, minutes, seconds

def print_summary(start, epoch, epochs, loss_train, loss_valid):
    end=time.time()
    hours, minutes, seconds =time_since(start, end)
    summary=f'{round(hours)}:{round(minutes)}:{round(seconds,2)} ({epoch + 1} {round((epoch + 1) / epochs * 100,2)}%) ' \
        f'loss train: {round(loss_train, 3)} loss valid: {round(loss_valid, 3)}'
    print(summary)

