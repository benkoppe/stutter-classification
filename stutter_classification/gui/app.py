import sys
from functools import partial

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QVBoxLayout,
    QWidget,
    QTextEdit,
    QSpacerItem,
    QSizePolicy,
    QSlider,
)
from qt_material import apply_stylesheet

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neural_network import MLPClassifier

from stutter_classification.models.base.stutter_model import StutterModel, TYPE_LABELS
from stutter_classification.models import SingleFeatureModel, AllFeaturesModel

from recorder import Recorder
from utils import make_labeled_combo_box, make_label, make_styled_label

APP_TITLE = "Stutter Detector"
WINDOW_GEOMETRY = (300, 300, 425, 725)

BUTTON_START_LABEL = "Start Recording"
BUTTON_STOP_LABEL = "Stop Recording"

STUTTER_DETECTION_LABEL = "Stutter Detected!"

TRANSCRIPTION_PLACEHOLDER = "Transcription will appear here..."
TRANSCRIPTION_BOX_SIZE = (350, 300)

SEPARATION_SPACER_HEIGHT = 10

MODEL_TYPE_OPTIONS = {
    "Single Label Model": SingleFeatureModel,
    "All Labels Model": AllFeaturesModel,
}
UNDERLYING_MODEL_OPTIONS = {
    "DecisionTreeClassifier": partial(DecisionTreeClassifier, criterion="gini"),
    "RandomForestClassifier": RandomForestClassifier,
    "GradientBoostingClassifier": GradientBoostingClassifier,
    "SVC": SVC,
    "GaussianNB": GaussianNB,
    "Neural Network": MLPClassifier,
}

MIN_N_MFCC = 5
MAX_N_MFCC = 21
DEFAULT_N_MFCC = 13

N_MFCC_LABEL = "Number of Feature Extraction (MFCC) Vectors: "
UNDERLYING_MODEL_LABEL = "Underlying Model: "
MODEL_TYPE_LABEL = "Model Type: "
FEATURE_TYPE_LABEL = "Label Name: "

MODEL_SCORE_LABEL_PREFIX = "Model Test Score: "


class SpeechApp(QMainWindow):
    recorder: Recorder

    n_mfccs = DEFAULT_N_MFCC
    underlying_model_type = list(UNDERLYING_MODEL_OPTIONS.values())[0]
    model_type = list(MODEL_TYPE_OPTIONS.values())[0]
    feature_type = TYPE_LABELS[0]

    def __init__(self):
        super().__init__()

        self.recorder = Recorder()
        self.recorder.update_transcription_signal.connect(self.update_transcription)
        self.recorder.update_test_score.connect(self.update_score_label)
        self.recorder.update_prediction.connect(self.update_prediction)

        self.initUI()
        self.update_model()

    def initUI(self):
        # Main Widget
        self.main_widget = QWidget(self)
        self.setCentralWidget(self.main_widget)

        # Layout
        layout = QVBoxLayout()
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        # Stutter detection label
        self.stutter_detection_label = make_styled_label(
            STUTTER_DETECTION_LABEL, self, font_size=30, color="red"
        )
        self.stutter_detection_label.setVisible(False)
        layout.addWidget(self.stutter_detection_label)

        # Button to start/stop recording
        self.btn = QPushButton(BUTTON_START_LABEL, self)
        self.btn.clicked.connect(self.toggle_recording)
        layout.addWidget(self.btn)

        # Label to display transcription
        self.transcription_text_edit = QTextEdit(self)
        self.transcription_text_edit.setReadOnly(True)
        self.transcription_text_edit.setPlaceholderText(TRANSCRIPTION_PLACEHOLDER)
        self.transcription_text_edit.setFixedSize(*TRANSCRIPTION_BOX_SIZE)
        layout.addWidget(self.transcription_text_edit)

        # Spacer
        layout.addItem(
            QSpacerItem(
                0,
                SEPARATION_SPACER_HEIGHT,
                QSizePolicy.Policy.Fixed,
                QSizePolicy.Policy.Fixed,
            )
        )

        self.n_mfcc_label = make_label(f"{N_MFCC_LABEL}{0}", self)
        self.n_mfcc_slider = QSlider(Qt.Orientation.Horizontal, self)
        self.n_mfcc_slider.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred
        )
        self.n_mfcc_slider.setMinimum(MIN_N_MFCC)
        self.n_mfcc_slider.setMaximum(MAX_N_MFCC)
        self.n_mfcc_slider.setValue(DEFAULT_N_MFCC)
        self.n_mfcc_slider.setTickInterval(1)
        self.n_mfcc_slider.setTickPosition(QSlider.TickPosition.TicksBothSides)
        self.n_mfcc_slider.valueChanged.connect(self.n_mfccs_changed)

        layout.addWidget(self.n_mfcc_label)
        layout.addWidget(self.n_mfcc_slider)

        # Underlying classifier labeled combo box
        self.underlying_model_label, self.underlying_model_combo = (
            make_labeled_combo_box(
                UNDERLYING_MODEL_LABEL, list(UNDERLYING_MODEL_OPTIONS.keys()), self
            )
        )
        self.underlying_model_combo.currentTextChanged.connect(
            self.underlying_model_type_changed
        )
        layout.addWidget(self.underlying_model_label)
        layout.addWidget(self.underlying_model_combo)

        # Model type labeled combo box
        self.model_type_label, self.model_type_combo = make_labeled_combo_box(
            MODEL_TYPE_LABEL, list(MODEL_TYPE_OPTIONS.keys()), self
        )
        self.model_type_combo.currentTextChanged.connect(self.model_type_changed)
        layout.addWidget(self.model_type_label)
        layout.addWidget(self.model_type_combo)

        # Feature type labeled combo box (depends on model type selection)
        self.feature_type_label, self.feature_type_combo = make_labeled_combo_box(
            FEATURE_TYPE_LABEL, TYPE_LABELS, self
        )
        self.feature_type_combo.currentTextChanged.connect(self.feature_type_changed)
        layout.addWidget(self.feature_type_label)
        layout.addWidget(self.feature_type_combo)

        # Score labels
        self.score_label = make_styled_label(
            f"{MODEL_SCORE_LABEL_PREFIX}0.0", self, font_size=13
        )
        layout.addWidget(self.score_label)

        self.main_widget.setLayout(layout)

        # Window settings
        self.setGeometry(*WINDOW_GEOMETRY)
        self.setWindowTitle(APP_TITLE)

    def toggle_recording(self):
        if not self.recorder.recording:
            self.btn.setText(BUTTON_STOP_LABEL)
            self.recorder.start_recording()
        else:
            self.btn.setText(BUTTON_START_LABEL)
            self.recorder.stop_recording()

    def update_transcription(self, text):
        if text == "":
            self.transcription_text_edit.clear()
        else:
            self.transcription_text_edit.append(text)

    def n_mfccs_changed(self, n_mfccs):
        self.n_mfccs = n_mfccs
        self.update_model()

    def underlying_model_type_changed(self, model_type_str):
        self.underlying_model_type = UNDERLYING_MODEL_OPTIONS[model_type_str]
        self.update_model()

    def model_type_changed(self, model_type_str):
        self.model_type = MODEL_TYPE_OPTIONS[model_type_str]

        if self.model_type == SingleFeatureModel:
            self.feature_type_label.setVisible(True)
            self.feature_type_combo.setVisible(True)
        else:
            self.feature_type_label.setVisible(False)
            self.feature_type_combo.setVisible(False)

        self.update_model()

    def feature_type_changed(self, feature_type_str):
        self.feature_type = feature_type_str
        self.update_model()

    def update_model(self):
        self.n_mfcc_label.setText(f"{N_MFCC_LABEL}{self.n_mfccs}")
        self.recorder.set_model(
            self.underlying_model_type, self.model_type, self.feature_type, self.n_mfccs
        )

    def update_score_label(self, score):
        self.score_label.setText(f"{MODEL_SCORE_LABEL_PREFIX}{score * 100:.2f}")

    def update_prediction(self, prediction):
        if prediction == "1":
            self.stutter_detected()
        elif prediction == "0":
            self.stutter_not_detected()
        elif prediction != "NoStutteredWords" and prediction != "NaturalPause":
            self.stutter_detected()
        else:
            self.stutter_not_detected()

    def stutter_detected(self):
        self.stutter_detection_label.setVisible(True)

    def stutter_not_detected(self):
        self.stutter_detection_label.setVisible(False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    apply_stylesheet(app, theme="dark_teal.xml")
    ex = SpeechApp()

    # ensure recording is stopped when closing the app
    app.aboutToQuit.connect(ex.recorder.stop_recording)
    ex.show()
    sys.exit(app.exec())
