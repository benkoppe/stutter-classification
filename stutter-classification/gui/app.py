import sys
from PyQt6.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget
from qt_material import apply_stylesheet

from recorder import Recorder


class AudioRecorder(QMainWindow):
    recorder = Recorder()

    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        # Main Widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Layout and button
        layout = QVBoxLayout()
        self.btn = QPushButton("Start Recording", self)
        self.btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.btn)

        self.main_widget.setLayout(layout)

        # Window settings
        self.setGeometry(300, 300, 400, 600)
        self.setWindowTitle("Audio Recorder")

    def toggle_recording(self):
        if self.btn.text() == "Start Recording":
            self.btn.setText("Stop Recording")
            self.recorder.start_recording()
        else:
            self.btn.setText("Start Recording")
            self.recorder.stop_recording()


if __name__ == "__main__":
    app = QApplication(sys.argv)

    apply_stylesheet(app, theme="dark_teal.xml")

    # ensure recording is stopped when closing the app
    app.aboutToQuit.connect(AudioRecorder.recorder.stop_recording)

    ex = AudioRecorder()
    ex.show()
    sys.exit(app.exec())
