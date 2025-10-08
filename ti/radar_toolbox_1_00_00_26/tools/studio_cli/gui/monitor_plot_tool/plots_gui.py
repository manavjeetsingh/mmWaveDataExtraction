import plots
import sys
from tkinter import Tk, Label, Button

class MonitorPlotGUI:

    def __init__(self, root, path=None):

        if path is not None:
            self.path = path
        else:
            with open("c_python_mailbox.txt", "r") as f:
                self.path = f.readline()
        
        self.root = root
        root.title("Monitor Plot GUI")
        root.geometry("300x400")
        
        self.rlMonPllConVoltRepButton = Button(root, text="PLL Control Voltage Monitor", command= lambda: plots.rlMonPllConVoltRepDataPlot(self.path), padx=10, pady=10)
        self.rlMonPllConVoltRepButton.pack(padx=10, pady=10)

        self.rlMonRxIfStageRepButton = Button(root, text="RX IF Stage Monitor", command= lambda: plots.rlMonRxIfStageRepDataPlot(self.path), padx=10, pady=10)
        self.rlMonRxIfStageRepButton.pack(padx=10, pady=10)

        self.rlMonSynthFreqRepButton = Button(root, text="Synth Frequency Monitor", command= lambda: plots.rlMonSynthFreqRepDataPlot(self.path), padx=10, pady=10)
        self.rlMonSynthFreqRepButton.pack(padx=10, pady=10)

        self.rlMonTempReportButton = Button(root, text="Temperature Monitor", command= lambda: plots.rlMonTempReportDataPlot(self.path), padx=10, pady=10)
        self.rlMonTempReportButton.pack(padx=10, pady=10)

        self.rlMonTxBallBreakRepButton = Button(root, text="TX Ballbreak Monitor", command= lambda: plots.rlMonTxBallBreakRepDataPlot(self.path), padx=10, pady=10)
        self.rlMonTxBallBreakRepButton.pack(padx=10, pady=10)

        self.rlMonTxPowRepButton = Button(root, text="TX Power Monitor", command= lambda: plots.rlMonTxPowRepDataPlot(self.path), padx=10, pady=10)
        self.rlMonTxPowRepButton.pack(padx=10, pady=10)
    

if __name__ == "__main__":
    root = Tk()
    if(len(sys.argv) == 2):
        MonitorPlotGUI(root, sys.argv[1])
    else:
        MonitorPlotGUI(root)
    root.mainloop()