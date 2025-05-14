



# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    from pypf.pfnet import PFnet
    from pypf.proximity import Proximity
    import pypf as pf
    psyp = Proximity("data/psy.prx.xlsx")
    psyp.prxprint()
    psy = PFnet(psyp)
    psy.netprint()
    psyq2 = PFnet(psyp,q=2)
    psyq2.netprint()
    biop = Proximity("data/bio.prx.xlsx")
    bio = PFnet(biop)
    bio.netprint()

    from pypf.utility import discorr
    cor = discorr(psyp.dismat,biop.dismat)
    print(f"Discorr: {cor}")
    sim = pf.netsim(psy.adjmat,bio.adjmat)
    print("Netsim: ", sim)

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
