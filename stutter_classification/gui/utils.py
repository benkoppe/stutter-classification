from PyQt6.QtWidgets import QComboBox, QLabel, QSizePolicy


def make_labeled_combo_box(name, options: list[str], parent):
    label = QLabel(name, parent)
    label.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Fixed)

    combo_box = QComboBox(parent)
    for option in options:
        combo_box.addItem(option)
    combo_box.setStyleSheet(
        """
        QComboBox {
            color: white;  /* Change text color to white */
        }
        """
    )
    return label, combo_box


def make_styled_label(name, parent, font_size=12, color="white"):
    label = make_label(name, parent)

    # set font size
    font = label.font()
    font.setPointSize(font_size)
    label.setFont(font)

    # set color
    label.setStyleSheet(
        f"""
        QLabel {{
            color: {color};  /* Change text color */
            font-size: {font_size}px;  /* Change font size */
        }}
        """
    )

    return label


def make_label(name, parent):
    label = QLabel(name, parent)
    label.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)

    return label
