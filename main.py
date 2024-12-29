import sys
import os
from PyQt5.QtWidgets import QApplication, QMainWindow, QMessageBox
from gui import Ui_MainWindow  # Replace 'gui' with the actual file name of your UI module

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.main_win = QMainWindow()
        self.uic = Ui_MainWindow()
        self.uic.setupUi(self.main_win)
        self.script_path = "classification.py"

        # Connect the button to the classification function
        self.uic.pushButton.clicked.connect(self.classification)

    def classification(self):
        try:
            # Run the classification script using os.system
            exit_code = os.system(f"python {self.script_path}")
            
            # Display message based on the exit code
            if exit_code == 0:
                QMessageBox.information(self, "Success", "The classification script executed successfully!")
            else:
                QMessageBox.warning(self, "Error", f"The script finished with a non-zero exit code: {exit_code}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"An error occurred: {e}")

    def show(self):
        self.main_win.show()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    main_win = MainWindow()
    main_win.show()
    sys.exit(app.exec())
