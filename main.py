# This is a sample Python script.
import numpy as np


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#TODO =

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    import pypf as pf
    prx = pf.Proximity()
    prx.prxprint()
    net1 = pf.PFnet(prx)
    net1.netprint()
    net2 = pf.PFnet(q=2)
    net2.netprint()
    net3 = pf.PFnet(q=2)

    from pypf.utility import discorr, netsim
    cor = discorr(prx.dismat,net3.dismat)
    print(f"Discorr: {cor}")
    sim = netsim(net1.adjmat,net3.adjmat)
    print(f"Netsim: {sim}")


# See PyCharm help at https://www.jetbrains.com/help/pycharm/
