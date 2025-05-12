# This is a sample Python script.
import numpy as np

from pypf import PFnet


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#TODO =

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')

    from pypf import PFnet, Proximity
    psyp = Proximity("data/psy.prx.xlsx")
    psyp.prxprint()
    psy = PFnet(psyp)
    psy.netprint()
    psyq2 = PFnet(psyp,q=2)
    psyq2.netprint()
    biop = Proximity("data/bio.prx.xlsx")
    bio = PFnet(biop)
    bio.netprint()

    from pypf.utility import discorr, netsim
    cor = discorr(psyp.dismat,biop.dismat)
    print(f"Discorr: {cor}")
    sim = netsim(psy.adjmat,bio.adjmat)
    print("Netsim: ", sim)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
