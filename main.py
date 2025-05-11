# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
#TODO = "prxcor netsim "

def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
    import pypf as pf
    prx = pf.Proximity()
    prx.prxprint()
    net = pf.PFnet(prx)
    net.netprint()
    net2 = pf.PFnet(q=2)
    net2.netprint()

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
