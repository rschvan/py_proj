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




        



