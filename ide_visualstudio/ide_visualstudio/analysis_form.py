
from tkinter import *
from tkinter import messagebox as tkMessageBox
import sys
import subprocess
import time

class Window(Frame):
    array_agr = ""

    def __init__(self, master=None):
        Frame.__init__(self, master)
        self.master = master
        self.init_window()

    def init_window(self):
        self.master.title("Statement Analysis Application")
        self.pack(fill=BOTH, expand=1)

        self.path_text = StringVar()
        self.neural_check = IntVar()

        # define labels in form
        lblPath = Label(self, text="Import path of files: ", fg="Steel Blue", bd=10, anchor="w")
        lblPath.grid(row=0, column=0, sticky=W)

        # define entries in form
        self.txtPath = Entry(self, width=100, textvariable=self.path_text)
        self.txtPath.grid(row=0, column=1, sticky=W)
        self.path_text.set("./data/data_test/")

        # define Checkbuttons in form
        self.chk_CNN = Radiobutton(self, text="CNN", variable=self.neural_check, font=("arial", 10), value=1)
        self.chk_CNN.grid(row=1, column=0, sticky=W)

        self.chk_RNN = Radiobutton(self, text="RNN", variable=self.neural_check, font=("arial", 10), value=2)
        self.chk_RNN.grid(row=2, column=0, sticky=W)

        self.chk_CNN_RNN_Pra = Radiobutton(self, text="CNN and RNN (parallel)", variable=self.neural_check, font=("arial", 10), value=3)
        self.chk_CNN_RNN_Pra.grid(row=3, column=0, sticky=W)

        self.chk_CNN_RNN_Seq = Radiobutton(self, text="CNN and RNN (sequence)", variable=self.neural_check, font=("arial", 10), value=4)
        self.chk_CNN_RNN_Seq.grid(row=4, column=0, sticky=W)

        self.chk_All_Model = Radiobutton(self, text="All Model", variable=self.neural_check,
                                           font=("arial", 10), value=5)
        self.chk_All_Model.grid(row=5, column=0, sticky=W)

        # define buttons in form
        # btnOpenPath = Button(self, text="...", command=self.client_exit)
        # btnOpenPath.grid(row=0, column=2, sticky=W)

        btnRun = Button(self, text="Run Analysis", command=self.run_Analysis)
        btnRun.grid(row=6, column=0, sticky=W)

    def run_Analysis(self):
        array_agr = ""
        if self.neural_check.get() == 1:
            array_agr = "CNN"
        elif self.neural_check.get() == 2:
            array_agr = "RNN"
        elif self.neural_check.get() == 3:
            array_agr = "Pra"
        elif self.neural_check.get() == 4:
            array_agr = "Seq"
        else:
            array_agr = "CNN,RNN,Pra,Seq"

        if array_agr == "":
            tkMessageBox.showinfo("Warning", "Please check for model neural network to run analysis.")
        else:
            list_arg = list(map(str, array_agr.split(",")))
            log_file = "result_" + str(int(time.time()))
            folder_data = self.path_text.get()
            print(folder_data)
            for i in range(len(list_arg)):
                arg_value = list_arg[i]
                file = "./train_orther.py"
                if arg_value == "Pra":
                    file = "./train.py"
                subprocess.call([sys.executable, file, folder_data, log_file, str(arg_value)])
                
def center(win):
    """
    centers a tkinter window
    :param win: the root or Toplevel window to center
    """
    win.update_idletasks()
    width = win.winfo_width()
    frm_width = win.winfo_rootx() - win.winfo_x()
    win_width = width + 2 * frm_width
    height = win.winfo_height()
    titlebar_height = win.winfo_rooty() - win.winfo_y()
    win_height = height + titlebar_height + frm_width
    x = win.winfo_screenwidth() // 2 - win_width // 2
    y = win.winfo_screenheight() // 2 - win_height // 2
    win.geometry('{}x{}+{}+{}'.format(width, height, x, y))
    win.deiconify()

def main():
    # Create window object
    root = Tk()
    root.geometry("800x400+0+0")

    center(root)

    app = Window(root)
    app.mainloop()
    
if __name__ == '__main__':
    main()