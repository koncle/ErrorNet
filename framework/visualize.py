import matplotlib.pyplot as plt

def show_graph(img, pred, mask):
    plt.subplot(131)
    plt.imshow(img, cmap="gray")
    plt.subplot(132)
    plt.imshow(pred, cmap="gray")
    plt.subplot(133)
    plt.imshow(mask, cmap="gray")
    plt.show()
