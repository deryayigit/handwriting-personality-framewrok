if __name__ == "__main__" and __package__ is None:
    import sys
    from pathlib import Path
    sys.path.append(str(Path(__file__).resolve().parents[1]))

import random
import sys
from pathlib import Path
from typing import Dict, Any, Optional, List

import numpy as np

from PyQt6.QtCore import Qt, QThread, pyqtSignal
from PyQt6.QtGui import QPixmap
from PyQt6.QtWidgets import (
    QApplication, QWidget, QLabel, QPushButton, QFileDialog,
    QVBoxLayout, QHBoxLayout, QGroupBox, QProgressBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QMessageBox,
    QComboBox, QSplitter, QFrame, QSizePolicy, QDialog, QLineEdit
)

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from matplotlib.colors import LinearSegmentedColormap, Normalize

from gui.controller import AppController


APP_STYLE = """
QWidget {
    background-color: #07111f;
    color: #e8e8f0;
    font-family: Segoe UI;
    font-size: 10px;
}
QFrame { background-color: transparent; }
QGroupBox {
    border: 1px solid #24364f;
    border-radius: 8px;
    margin-top: 10px;
    padding: 8px;
    background-color: #0b1626;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 5px;
    color: #b66cff;
    font-size: 10px;
    font-weight: 600;
}
QLabel { color: #e8e8f0; font-size: 10px; }
QPushButton {
    background-color: #6d35c9;
    color: white;
    border: none;
    border-radius: 6px;
    padding: 4px 7px;
    font-size: 10px;
    font-weight: 600;
    min-height: 22px;
}
QPushButton:hover { background-color: #8049df; }
QPushButton:disabled {
    background-color: #2a3140;
    color: #8a8f9c;
}
QComboBox {
    background-color: #0f1d30;
    color: white;
    border: 1px solid #24364f;
    border-radius: 5px;
    padding: 3px;
    font-size: 10px;
    min-height: 22px;
}
QTableWidget {
    background-color: #0b1626;
    alternate-background-color: #101f35;
    color: #e8e8f0;
    gridline-color: #24364f;
    border: 1px solid #24364f;
    border-radius: 6px;
    font-size: 9px;
}
QHeaderView::section {
    background-color: #121f35;
    color: #b66cff;
    border: 1px solid #24364f;
    padding: 3px;
    font-size: 9px;
    font-weight: 600;
}
QProgressBar {
    border: 1px solid #24364f;
    border-radius: 5px;
    background-color: #111d30;
    height: 10px;
}
QProgressBar::chunk {
    background-color: #7c4dff;
    border-radius: 5px;
}
QSplitter::handle { background-color: #13233b; }
"""


MODEL_LABEL_ORDER = [
    "Agreeableness",
    "Conscientiousness",
    "Extraversion",
    "Neuroticism",
    "Openness"
]

OCEAN_ORDER = [
    "Openness",
    "Conscientiousness",
    "Extraversion",
    "Agreeableness",
    "Neuroticism"
]

OCEAN_SHORT = ["O", "C", "E", "A", "N"]


class AnalysisWorker(QThread):
    result_ready = pyqtSignal(dict)
    error_ready = pyqtSignal(str)

    def __init__(
        self,
        controller: AppController,
        model_path: str,
        image_path: str,
        use_preprocessing: bool,
        task_type: str
    ):
        super().__init__()
        self.controller = controller
        self.model_path = model_path
        self.image_path = image_path
        self.use_preprocessing = use_preprocessing
        self.task_type = task_type

    def run(self):
        try:
            result = self.controller.analyze_image(
                model_path=self.model_path,
                image_path=self.image_path,
                use_preprocessing=self.use_preprocessing,
                task_type=self.task_type
            )
            self.result_ready.emit(result)

        except Exception as error:
            self.error_ready.emit(str(error))


class TrainingChart(FigureCanvas):
    def __init__(self, width=3.6, height=1.6):
        self.fig = Figure(figsize=(width, height), dpi=100)
        self.fig.patch.set_facecolor("#0b1626")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.setMaximumHeight(210)
        self.setMinimumWidth(300)
        self.setStyleSheet("background-color: #0b1626;")
        self.plot_history([])

    def plot_history(self, history: Optional[List[Dict[str, Any]]]):
        self.fig.clear()
        self.fig.patch.set_facecolor("#0b1626")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0b1626")

        if not history:
            self.ax.set_axis_off()
            self.ax.text(
                0.5, 0.5, "No training history",
                color="white", ha="center", va="center",
                transform=self.ax.transAxes
            )
            self.draw()
            return

        epochs = [
            item.get("global_epoch", index + 1)
            for index, item in enumerate(history)
        ]

        fields = [
            ("train_loss", "Train Loss"),
            ("val_loss", "Val Loss"),
            ("train_accuracy", "Train Acc"),
            ("val_accuracy", "Val Acc"),
            ("val_macro_f1", "Val F1"),
            ("train_rmse", "Train RMSE"),
            ("val_rmse", "Val RMSE"),
            ("val_mae", "Val MAE")
        ]

        plotted = False

        for key, label in fields:
            values = [item.get(key) for item in history]

            if all(value is not None for value in values):
                self.ax.plot(epochs, values, label=label, linewidth=1.2)
                plotted = True

        if not plotted:
            self.ax.set_axis_off()
            self.ax.text(
                0.5, 0.5, "No plottable metrics",
                color="white", ha="center", va="center",
                transform=self.ax.transAxes
            )
            self.draw()
            return

        self.ax.set_title("Training Progress", color="white", fontsize=8)
        self.ax.tick_params(axis="both", colors="white", labelsize=6)
        self.ax.grid(True, linestyle="--", alpha=0.22)

        for spine in self.ax.spines.values():
            spine.set_color("#3a4d68")

        legend = self.ax.legend(
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            facecolor="#101f35",
            edgecolor="#3a4d68",
            fontsize=5
        )

        for text in legend.get_texts():
            text.set_color("white")

        self.fig.subplots_adjust(left=0.13, right=0.72, top=0.82, bottom=0.20)
        self.draw()


class ConfusionMatrixCanvas(FigureCanvas):
    def __init__(self, width=8.8, height=8.8):
        self.fig = Figure(figsize=(width, height), dpi=100)
        self.fig.patch.set_facecolor("#0b1626")
        self.ax = self.fig.add_subplot(111)
        super().__init__(self.fig)

        self.setMinimumHeight(610)
        self.setSizePolicy(
            QSizePolicy.Policy.Expanding,
            QSizePolicy.Policy.Expanding
        )
        self.setStyleSheet("background-color: #0b1626;")
        self.plot_matrix(None, None)

    def _reorder_to_ocean(self, matrix, labels):
        matrix_np = np.array(matrix)

        if matrix_np.shape[0] != len(labels):
            return matrix_np, labels

        indices = []

        for name in OCEAN_ORDER:
            if name in labels:
                indices.append(labels.index(name))

        if len(indices) != len(labels):
            return matrix_np, labels

        reordered = matrix_np[np.ix_(indices, indices)]
        return reordered, OCEAN_ORDER

    def plot_matrix(
        self,
        matrix: Optional[List[List[int]]],
        labels: Optional[List[str]],
        title: str = "Global Confusion Matrix",
        empty_message: str = "Confusion matrix not found"
    ):
        self.fig.clear()
        self.fig.patch.set_facecolor("#0b1626")
        self.ax = self.fig.add_subplot(111)
        self.ax.set_facecolor("#0b1626")

        if not matrix or not labels:
            self.ax.set_axis_off()
            self.ax.text(
                0.5, 0.55, title,
                color="#d8b4ff", fontsize=16,
                ha="center", va="center",
                transform=self.ax.transAxes
            )
            self.ax.text(
                0.5, 0.45, empty_message,
                color="white", fontsize=13,
                ha="center", va="center",
                transform=self.ax.transAxes
            )
            self.draw()
            return

        matrix_np, ordered_labels = self._reorder_to_ocean(matrix, labels)

        purple_map = LinearSegmentedColormap.from_list(
            "custom_purple",
            ["#f8f4ff", "#e9d5ff", "#c084fc", "#7e22ce", "#3b0764"]
        )

        max_value = matrix_np.max() if matrix_np.size else 1
        norm = Normalize(vmin=0, vmax=max_value)

        image = self.ax.imshow(
            matrix_np,
            cmap=purple_map,
            norm=norm,
            interpolation="nearest"
        )

        tick_labels = OCEAN_SHORT if ordered_labels == OCEAN_ORDER else ordered_labels

        self.ax.set_xticks(np.arange(len(ordered_labels)))
        self.ax.set_yticks(np.arange(len(ordered_labels)))

        self.ax.set_xticklabels(
            tick_labels,
            color="white",
            fontsize=14,
            fontweight="bold"
        )

        self.ax.set_yticklabels(
            tick_labels,
            color="white",
            fontsize=14,
            fontweight="bold"
        )

        self.ax.set_xlabel(
            "Predicted Label",
            color="white",
            fontsize=13,
            fontweight="bold",
            labelpad=10
        )

        self.ax.set_ylabel(
            "True Label",
            color="white",
            fontsize=13,
            fontweight="bold",
            labelpad=10
        )

        self.ax.set_title(
            title,
            color="white",
            fontsize=18,
            fontweight="bold",
            pad=14
        )

        self.ax.tick_params(axis="both", which="major", colors="white")

        self.ax.set_xticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)
        self.ax.set_yticks(np.arange(-0.5, len(ordered_labels), 1), minor=True)

        self.ax.grid(
            which="minor",
            color="#263955",
            linestyle="-",
            linewidth=1.2
        )

        self.ax.tick_params(which="minor", bottom=False, left=False)

        for row in range(matrix_np.shape[0]):
            for col in range(matrix_np.shape[1]):
                value = int(matrix_np[row, col])
                text_color = "white" if value >= max_value * 0.45 else "#c084fc"

                self.ax.text(
                    col, row, str(value),
                    ha="center", va="center",
                    color=text_color,
                    fontsize=16,
                    fontweight="bold"
                )

        for spine in self.ax.spines.values():
            spine.set_color("#3a4d68")
            spine.set_linewidth(1.2)

        colorbar = self.fig.colorbar(
            image,
            ax=self.ax,
            fraction=0.035,
            pad=0.04
        )
        colorbar.ax.tick_params(colors="white", labelsize=9)
        colorbar.outline.set_edgecolor("#3a4d68")

        self.ax.set_aspect("equal")
        self.fig.subplots_adjust(left=0.16, right=0.90, top=0.88, bottom=0.15)
        self.draw()


class HandwritingDashboard(QWidget):
    def __init__(self):
        super().__init__()

        self.controller = AppController()

        self.setWindowTitle("El Yazısından Kişilik Analizi")
        self.resize(1700, 930)
        self.setMinimumSize(1350, 780)

        self.image_path = None
        self.processed_image_path = None
        self.current_expected_class = None
        self.worker = None

        self.current_task_type = "classification"

        self.demo_image_paths = {}
        self.demo_indices = {}
        self.comparison_rows = []

        # Aktif oturum boyunca tutulan geçici tahmin karşılaştırma verileri.
        # Dosyaya, JSON'a veya veritabanına yazılmaz; uygulama kapanınca silinir.
        self.session_prediction_results = {}
        self.session_ipip_results = {}
        self.current_image_key = None
        self.current_image_display_name = None
        self.current_analysis_type = None

        self.analysis_experiment = self.controller.get_experiment_by_id(
            self.controller.default_analysis_experiment
        )
        self.analysis_checkpoint = self.controller.get_default_analysis_checkpoint()
        self.analysis_model_path = (
            self.analysis_checkpoint["path"]
            if self.analysis_checkpoint
            else None
        )

        self.matrix_experiment = None
        self.matrix_checkpoint = None

        self._build_ui()
        self._load_experiment_values()
        self._load_default_sample_dataset()
        self._update_active_model_text()

    def _build_ui(self):
        main_layout = QVBoxLayout()
        main_layout.setContentsMargins(8, 8, 8, 8)
        main_layout.setSpacing(6)

        title = QLabel("El Yazısından Kişilik Analizi")
        title.setStyleSheet("font-size: 17px; font-weight: bold; color: white;")
        main_layout.addWidget(title)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self._build_left_panel())
        splitter.addWidget(self._build_right_panel())
        splitter.setSizes([760, 940])

        main_layout.addWidget(splitter, 1)
        self.setLayout(main_layout)

    def _build_left_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(5)

        sample_box = QGroupBox("Test Sonuçları")
        sample_layout = QVBoxLayout()
        sample_layout.setSpacing(4)

        random_button = QPushButton("Test Kümesinden Rastgele Örnek Seçiniz")
        random_button.setFixedHeight(26)
        random_button.clicked.connect(self._load_random_test_sample)
        sample_layout.addWidget(random_button)

        ocean_row = QHBoxLayout()
        self.sample_buttons = {}

        ocean_buttons = [
            ("O", "Openness"),
            ("C", "Conscientiousness"),
            ("E", "Extraversion"),
            ("A", "Agreeableness"),
            ("N", "Neuroticism")
        ]

        for short_name, class_name in ocean_buttons:
            button = QPushButton(short_name)
            button.setFixedHeight(26)
            button.clicked.connect(
                lambda checked=False, c=class_name: self._load_next_sample(c)
            )
            self.sample_buttons[class_name] = button
            ocean_row.addWidget(button)

        sample_layout.addLayout(ocean_row)

        self.sample_result_label = QLabel("Son Örneğin Sonucu: —")
        self.sample_result_label.setWordWrap(True)
        self.sample_result_label.setStyleSheet("font-weight: bold; color: #d8b4ff;")
        sample_layout.addWidget(self.sample_result_label)

        sample_box.setLayout(sample_layout)

        image_box = QGroupBox("El Yazısı Analizi")
        image_layout = QVBoxLayout()
        image_layout.setSpacing(6)

        self.browse_button = QPushButton("Görsel Seçiniz")
        self.browse_button.setFixedHeight(28)
        self.browse_button.clicked.connect(self._browse_image)
        image_layout.addWidget(self.browse_button)

        image_row = QHBoxLayout()

        original_box = QGroupBox("Gerçek Görüntü")
        original_layout = QVBoxLayout()

        self.original_image_label = QLabel("Görüntü Seçilmedi")
        self.original_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.original_image_label.setFixedHeight(330)
        self.original_image_label.setStyleSheet(
            "background-color: #101f35; border: 1px solid #24364f; border-radius: 6px;"
        )

        self.analyze_original_button = QPushButton("Gerçek Görüntü Analizi")
        self.analyze_original_button.setFixedHeight(28)
        self.analyze_original_button.clicked.connect(self._analyze_original_image)

        original_layout.addWidget(self.original_image_label)
        original_layout.addWidget(self.analyze_original_button)
        original_box.setLayout(original_layout)

        processed_box = QGroupBox("Ön İşlenmiş Görüntü")
        processed_layout = QVBoxLayout()

        self.processed_image_label = QLabel("Ön İşleme Uygulanmadı")
        self.processed_image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.processed_image_label.setFixedHeight(330)
        self.processed_image_label.setStyleSheet(
            "background-color: #101f35; border: 1px solid #24364f; border-radius: 6px;"
        )

        self.preprocess_button = QPushButton("Ön İşleme Analizi")
        self.preprocess_button.setFixedHeight(28)
        self.preprocess_button.clicked.connect(self._analyze_preprocessed_image)

        processed_layout.addWidget(self.processed_image_label)
        processed_layout.addWidget(self.preprocess_button)
        processed_box.setLayout(processed_layout)

        image_row.addWidget(original_box, 1)
        image_row.addWidget(processed_box, 1)

        image_layout.addLayout(image_row)
        image_box.setLayout(image_layout)

        trait_box = QGroupBox("Kişilik Tahminleri")
        trait_layout = QVBoxLayout()
        trait_layout.setSpacing(7)

        self.trait_bars = {}
        self.trait_value_labels = {}

        for trait in OCEAN_ORDER:
            row = QHBoxLayout()

            name_label = QLabel(trait)
            name_label.setMinimumWidth(130)

            bar = QProgressBar()
            bar.setRange(0, 100)
            bar.setValue(0)
            bar.setTextVisible(False)

            value_label = QLabel("—")
            value_label.setMinimumWidth(50)
            value_label.setAlignment(Qt.AlignmentFlag.AlignRight)

            self.trait_bars[trait] = bar
            self.trait_value_labels[trait] = value_label

            row.addWidget(name_label)
            row.addWidget(bar, 1)
            row.addWidget(value_label)

            trait_layout.addLayout(row)

        self.active_model_label = QLabel("Seçili Model: —")
        self.active_model_metric_label = QLabel("Performans: —")

        self.active_model_label.setWordWrap(True)
        self.active_model_metric_label.setWordWrap(True)

        self.active_model_label.setStyleSheet("font-weight: bold; color: #d8b4ff;")
        self.active_model_metric_label.setStyleSheet("font-weight: bold; color: #58d68d;")

        trait_layout.addWidget(self.active_model_label)
        trait_layout.addWidget(self.active_model_metric_label)

        trait_box.setLayout(trait_layout)

        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(6)

        self.training_chart = TrainingChart(width=3.4, height=1.8)

        bottom_row.addWidget(trait_box, 3)
        bottom_row.addWidget(self.training_chart, 2)

        layout.addWidget(sample_box, 1)
        layout.addWidget(image_box, 5)
        layout.addLayout(bottom_row, 2)

        return panel

    def _build_right_panel(self):
        panel = QFrame()
        layout = QVBoxLayout(panel)
        layout.setSpacing(6)

        matrix_box = QGroupBox("Deney Sonuçları")
        matrix_layout = QVBoxLayout()
        matrix_layout.setSpacing(6)

        experiment_row = QHBoxLayout()

        experiment_label = QLabel("Deney")
        experiment_label.setFixedWidth(80)

        self.experiment_combo = QComboBox()
        self.experiment_combo.currentIndexChanged.connect(
            self._load_matrix_experiment_from_combo
        )

        experiment_row.addWidget(experiment_label)
        experiment_row.addWidget(self.experiment_combo, 1)

        matrix_layout.addLayout(experiment_row)

        self.confusion_canvas = ConfusionMatrixCanvas(width=8.8, height=8.8)
        matrix_layout.addWidget(self.confusion_canvas, 8)

        self.metric_summary_label = QLabel(
            "Precision: — | Recall: — | Total: — | Correct: —"
        )
        self.metric_summary_label.setWordWrap(True)
        self.metric_summary_label.setStyleSheet(
            "font-size: 11px; font-weight: bold; color: #d8b4ff;"
        )

        matrix_layout.addWidget(self.metric_summary_label)

        self.open_comparison_button = QPushButton("Deney Karşılaştırmalarını Aç")
        self.open_comparison_button.setFixedHeight(32)
        self.open_comparison_button.clicked.connect(self._open_comparison_dialog)

        matrix_layout.addWidget(self.open_comparison_button)

        self.open_prediction_comparison_button = QPushButton("Tahmin Karşılaştırmalarını Aç")
        self.open_prediction_comparison_button.setFixedHeight(32)
        self.open_prediction_comparison_button.clicked.connect(
            self._open_prediction_comparison_dialog
        )

        matrix_layout.addWidget(self.open_prediction_comparison_button)

        matrix_box.setLayout(matrix_layout)

        layout.addWidget(matrix_box, 1)

        return panel

    def _make_pixmap_for_label(self, image_path: str, label: QLabel) -> QPixmap:
        pixmap = QPixmap(image_path)

        if pixmap.isNull():
            return QPixmap()

        return pixmap.scaled(
            label.width(),
            label.height(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

    def _load_experiment_values(self):
        experiments = self.controller.list_experiments()

        self.experiment_combo.blockSignals(True)
        self.experiment_combo.clear()

        for experiment in experiments:
            if not experiment.get("run_exists"):
                continue

            task = experiment.get("task", "classification")

            text = (
                f"{experiment['experiment_id']} | "
                f"{task.upper()} | "
                f"{experiment.get('dataset_id')} | "
                f"{experiment.get('split_type')} | "
                f"CW={'ON' if experiment.get('class_weight') else 'OFF'} | "
                f"Balanced={'ON' if experiment.get('use_weighted_sampler') else 'OFF'} | "
                f"Aug={experiment.get('augmentation_strength', 'standard')}"
            )

            self.experiment_combo.addItem(text, experiment["experiment_id"])

        default_id = self.controller.default_analysis_experiment

        for index in range(self.experiment_combo.count()):
            if self.experiment_combo.itemData(index) == default_id:
                self.experiment_combo.setCurrentIndex(index)
                break

        self.experiment_combo.blockSignals(False)
        self._load_matrix_experiment_from_combo()

    def _load_matrix_experiment_from_combo(self):
        experiment_id = self.experiment_combo.currentData()

        if not experiment_id:
            return

        experiment = self.controller.get_experiment_by_id(experiment_id)

        if not experiment:
            return

        selected_checkpoint = self.controller.get_preferred_checkpoint(
            experiment_id=experiment_id,
            preferred_name="fold_1_model_best"
        )

        if not selected_checkpoint:
            self.matrix_experiment = experiment
            self.matrix_checkpoint = None
            self.analysis_experiment = None
            self.analysis_checkpoint = None
            self.analysis_model_path = None
            self.current_task_type = experiment.get("task", "classification")

            self.confusion_canvas.plot_matrix(
                None,
                None,
                title=f"{experiment_id}",
                empty_message="Model checkpoint bulunamadı."
            )

            self.metric_summary_label.setText(
                "Metric: —"
            )

            self._update_active_model_text()
            return

        self.matrix_experiment = experiment
        self.matrix_checkpoint = selected_checkpoint

        self.analysis_experiment = experiment
        self.analysis_checkpoint = selected_checkpoint
        self.analysis_model_path = selected_checkpoint["path"]
        self.current_task_type = experiment.get("task", "classification")

        if self.current_task_type == "classification":
            self._load_sample_dataset(experiment.get("dataset_id"))
        else:
            self.demo_image_paths = {}
            self.demo_indices = {}
            self.current_expected_class = None
            self.sample_result_label.setText("Regression deneylerinde sınıf tabanlı test örneği kullanılmaz.")

        self._update_active_model_text()
        self._update_metrics_and_confusion_matrix()
        self._add_current_configuration_to_comparison()

    def _load_default_sample_dataset(self):
        if not self.analysis_experiment:
            return

        if self.analysis_experiment.get("task", "classification") != "classification":
            return

        dataset_id = self.analysis_experiment.get("dataset_id")
        self._load_sample_dataset(dataset_id)

    def _metric_value(self, metrics: Dict[str, Any], keys: List[str], default="—"):
        for key in keys:
            if key in metrics and metrics[key] is not None:
                return metrics[key]

        return default

    def _format_metric(self, value):
        if isinstance(value, float):
            return f"{value:.4f}"

        return str(value)

    def _update_metrics_and_confusion_matrix(self):
        metrics = self.controller.get_metrics_for_checkpoint(self.matrix_checkpoint)
        best_metrics = self.controller.get_best_metrics_for_checkpoint(self.matrix_checkpoint)

        if metrics:
            self.training_chart.plot_history(metrics.get("history", []))
        else:
            self.training_chart.plot_history([])

        if not best_metrics:
            self.confusion_canvas.plot_matrix(
                None,
                None,
                title=f"{self.matrix_experiment['experiment_id']}",
                empty_message="Metrik dosyası bulunamadı."
            )
            self.metric_summary_label.setText("Metric: —")
            return

        task_type = self.matrix_experiment.get("task", "classification")

        if task_type == "regression":
            rmse = self._metric_value(best_metrics, ["rmse", "test_rmse", "avg_rmse", "RMSE"])
            mae = self._metric_value(best_metrics, ["mae", "test_mae", "avg_mae", "MAE"])
            pcc = self._metric_value(best_metrics, ["pcc", "pearson", "pearson_corr", "avg_pcc", "PCC"])
            dominant_acc = self._metric_value(
                best_metrics,
                ["dominant_accuracy", "dominant_acc", "test_dominant_accuracy", "avg_dominant_accuracy"]
            )

            self.confusion_canvas.plot_matrix(
                None,
                None,
                title=f"Regression Metrics - {self.matrix_experiment['experiment_id']}",
                empty_message="Regression deneylerinde Confusion Matrix kullanılmaz."
            )

            self.metric_summary_label.setText(
                f"RMSE: {self._format_metric(rmse)} | "
                f"MAE: {self._format_metric(mae)} | "
                f"PCC: {self._format_metric(pcc)} | "
                f"Dominant Acc: {self._format_metric(dominant_acc)}"
            )
            return

        matrix = best_metrics.get("confusion_matrix")
        report = best_metrics.get("classification_report", {})
        class_names = self._extract_class_names(report)

        self.confusion_canvas.plot_matrix(
            matrix,
            class_names,
            title=f"Global Confusion Matrix - {self.matrix_experiment['experiment_id']}"
        )

        accuracy = best_metrics.get("accuracy", "—")
        macro_f1 = best_metrics.get("macro_f1", "—")
        precision = best_metrics.get("macro_precision", "—")
        recall = best_metrics.get("macro_recall", "—")

        total, correct = self._compute_total_and_correct(matrix)

        self.metric_summary_label.setText(
            
            f"Precision: {precision} | Recall: {recall} | "
            f"Total: {total} | Correct: {correct}"
        )

    def _extract_class_names(self, report: Dict[str, Any]) -> List[str]:
        names = []

        for key in report.keys():
            if key in ["accuracy", "macro avg", "weighted avg"]:
                continue

            if key in MODEL_LABEL_ORDER:
                names.append(key)

        return names if names else MODEL_LABEL_ORDER

    def _compute_total_and_correct(self, matrix):
        if not matrix:
            return "—", "—"

        matrix_np = np.array(matrix)
        total = int(matrix_np.sum())
        correct = int(np.trace(matrix_np))

        return total, correct

    def _add_current_configuration_to_comparison(self):
        if not self.matrix_experiment or not self.matrix_checkpoint:
            return

        best_metrics = self.controller.get_best_metrics_for_checkpoint(
            self.matrix_checkpoint
        )

        row_data = self.controller.build_comparison_row(
            experiment=self.matrix_experiment,
            checkpoint=self.matrix_checkpoint,
            metrics=best_metrics
        )

        self.comparison_rows.append(row_data)

    def _open_comparison_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Deney Sonuçları")
        dialog.resize(1100, 500)
        dialog.setStyleSheet(APP_STYLE)

        layout = QVBoxLayout(dialog)

        table = QTableWidget()
        table.setColumnCount(12)
        table.setHorizontalHeaderLabels([
            "Exp", "Dataset", "Split", "CW", "Prep", "Balanced",
            "Aug", "Ckpt", "Acc/DomAcc", "F1", "Prec/RMSE", "Recall/MAE"
        ])

        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)
        table.setRowCount(len(self.comparison_rows))

        for row_index, row_data in enumerate(self.comparison_rows):
            values = [
                row_data["experiment_id"],
                row_data["dataset"],
                row_data["split_type"],
                row_data["class_weight"],
                row_data["preprocessing"],
                row_data["sampler"],
                row_data["augmentation"],
                row_data["checkpoint"],
                row_data["accuracy"],
                row_data["macro_f1"],
                row_data["precision"],
                row_data["recall"]
            ]

            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, column, item)

        clear_button = QPushButton("Sonuçları Temizle")
        clear_button.clicked.connect(lambda: self._clear_comparison_history(dialog))

        layout.addWidget(table)
        layout.addWidget(clear_button)

        dialog.exec()

    def _clear_comparison_history(self, dialog):
        self.comparison_rows.clear()
        dialog.close()

    def _format_trait_for_table(self, value):
        if value is None:
            return "—"

        try:
            return f"{float(value):.1f}%"
        except Exception:
            return "—"

    def _store_session_prediction_result(self, result: Dict[str, Any]):
        if not self.current_image_key:
            return

        analysis_type = self.current_analysis_type or "original"
        trait_values = self._extract_trait_values_from_result(result)

        entry = self.session_prediction_results.setdefault(
            self.current_image_key,
            {
                "display_name": self.current_image_display_name or "Seçili Görsel",
                "original": None,
                "enhanced": None
            }
        )

        entry["display_name"] = self.current_image_display_name or entry["display_name"]
        entry[analysis_type] = {
            "traits": trait_values,
            "predicted_class": result.get("predicted_class", "—"),
            "task_type": self.current_task_type,
            "experiment_id": (
                self.analysis_experiment.get("experiment_id", "—")
                if self.analysis_experiment
                else "—"
            ),
            "checkpoint": (
                self.analysis_checkpoint.get("name", "—")
                if self.analysis_checkpoint
                else "—"
            )
        }

    def _collect_prediction_comparison_rows(self):
        rows = []

        for image_key, entry in self.session_prediction_results.items():
            image_name = entry.get("display_name", "—")

            for analysis_type, label in [
                ("original", "Gerçek"),
                ("enhanced", "Ön İşlemeli")
            ]:
                result_data = entry.get(analysis_type)

                if not result_data:
                    continue

                traits = result_data.get("traits", {})

                rows.append({
                    "image_key": image_key,
                    "image_name": image_name,
                    "type": label,
                    "traits": traits
                })

            ipip_traits = self.session_ipip_results.get(image_key)

            rows.append({
                "image_key": image_key,
                "image_name": image_name,
                "type": "IPIP",
                "traits": ipip_traits or {}
            })

        return rows

    def _populate_prediction_comparison_table(self, table: QTableWidget):
        rows = self._collect_prediction_comparison_rows()
        table.setRowCount(len(rows))

        for row_index, row_data in enumerate(rows):
            values = [
                row_data["image_name"],
                row_data["type"],
            ]

            traits = row_data.get("traits", {})

            for trait in OCEAN_ORDER:
                values.append(self._format_trait_for_table(traits.get(trait)))

            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                table.setItem(row_index, column, item)

    def _open_prediction_comparison_dialog(self):
        if not self.session_prediction_results:
            QMessageBox.information(
                self,
                "Karşılaştırma Verisi Yok",
                "Henüz karşılaştırılacak analiz sonucu bulunmuyor. Önce gerçek veya ön işlemeli görüntü analizi yapmalısın."
            )
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("Tahmin Karşılaştırmaları")
        dialog.resize(1050, 560)
        dialog.setStyleSheet(APP_STYLE)

        main_layout = QHBoxLayout(dialog)
        main_layout.setSpacing(10)

        table = QTableWidget()
        table.setColumnCount(7)
        table.setHorizontalHeaderLabels([
            "Görsel", "Tür", "O", "C", "E", "A", "N"
        ])
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.setAlternatingRowColors(True)

        self._populate_prediction_comparison_table(table)

        ipip_box = QGroupBox("IPIP Sonucu Ekle / Güncelle")
        ipip_layout = QVBoxLayout()
        ipip_layout.setSpacing(8)

        image_combo = QComboBox()
        image_items = list(self.session_prediction_results.items())

        for image_key, entry in image_items:
            image_combo.addItem(entry.get("display_name", "—"), image_key)

        ipip_layout.addWidget(QLabel("Görsel"))
        ipip_layout.addWidget(image_combo)

        input_fields = {}

        for trait in OCEAN_ORDER:
            row = QHBoxLayout()
            label = QLabel(trait)
            label.setMinimumWidth(125)

            line_edit = QLineEdit()
            line_edit.setPlaceholderText("IPIP ham skor / yüzde")
            line_edit.setStyleSheet(
                "background-color: #0f1d30; color: white; "
                "border: 1px solid #24364f; border-radius: 5px; padding: 4px;"
            )

            input_fields[trait] = line_edit
            row.addWidget(label)
            row.addWidget(line_edit)
            ipip_layout.addLayout(row)

        info_label = QLabel(
            "Girilen IPIP değerleri toplamları 100 olacak şekilde kıyaslama amaçlı normalize edilir."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #d8b4ff;")
        ipip_layout.addWidget(info_label)

        def load_ipip_values_for_selected_image():
            image_key = image_combo.currentData()
            values = self.session_ipip_results.get(image_key, {})

            for trait, line_edit in input_fields.items():
                value = values.get(trait)
                line_edit.setText("" if value is None else f"{float(value):.1f}")

        def apply_ipip_values():
            image_key = image_combo.currentData()

            if not image_key:
                return

            raw_values = {}

            for trait, line_edit in input_fields.items():
                text = line_edit.text().strip().replace(",", ".")

                if not text:
                    QMessageBox.warning(
                        dialog,
                        "Eksik Değer",
                        "IPIP karşılaştırması için beş kişilik özelliğinin tamamını girmelisin."
                    )
                    return

                try:
                    raw_values[trait] = float(text)
                except ValueError:
                    QMessageBox.warning(
                        dialog,
                        "Geçersiz Değer",
                        f"{trait} için sayısal bir değer girmelisin."
                    )
                    return

            total = sum(raw_values.values())

            if total <= 0:
                QMessageBox.warning(
                    dialog,
                    "Geçersiz Toplam",
                    "IPIP değerlerinin toplamı sıfırdan büyük olmalıdır."
                )
                return

            normalized_values = {
                trait: (value / total) * 100.0
                for trait, value in raw_values.items()
            }

            self.session_ipip_results[image_key] = normalized_values

            for trait, line_edit in input_fields.items():
                line_edit.setText(f"{normalized_values[trait]:.1f}")

            self._populate_prediction_comparison_table(table)

        image_combo.currentIndexChanged.connect(load_ipip_values_for_selected_image)

        apply_button = QPushButton("IPIP Sonucunu Normalize Et ve Ekle")
        apply_button.clicked.connect(apply_ipip_values)
        ipip_layout.addWidget(apply_button)

        close_button = QPushButton("Kapat")
        close_button.clicked.connect(dialog.close)
        ipip_layout.addWidget(close_button)

        ipip_box.setLayout(ipip_layout)
        ipip_box.setFixedWidth(310)

        main_layout.addWidget(table, 3)
        main_layout.addWidget(ipip_box, 1)

        load_ipip_values_for_selected_image()
        dialog.exec()

    def _browse_image(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Seçili El Yazısı Örneği",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp)"
        )

        if not file_path:
            return

        self.current_expected_class = None
        self.processed_image_path = None

        prepared_path = self.controller.prepare_gui_image(file_path)
        self._load_image_to_gui(
            prepared_path,
            display_name=Path(file_path).name,
            image_key=str(Path(file_path).resolve())
        )

    def _load_image_to_gui(
        self,
        file_path: str,
        display_name: Optional[str] = None,
        image_key: Optional[str] = None
    ):
        self.image_path = file_path
        self.processed_image_path = None
        self.current_image_display_name = display_name or Path(file_path).name
        self.current_image_key = image_key or str(Path(file_path).resolve())


        pixmap = self._make_pixmap_for_label(
            image_path=file_path,
            label=self.original_image_label
        )

        self.original_image_label.setPixmap(pixmap)

        self.processed_image_label.setText("Ön İşleme Uygulanmadı")
        self.processed_image_label.setPixmap(QPixmap())

        for trait in OCEAN_ORDER:
            self.trait_bars[trait].setValue(0)
            self.trait_value_labels[trait].setText("—")

    def _analyze_original_image(self):
        if not self.image_path:
            QMessageBox.warning(
                self,
                "Görsel Bulunamadı",
                "Önce Görsel Seçiniz butonu ile bir el yazısı görseli seçmelisin."
            )
            return

        if not self.analysis_model_path or not Path(self.analysis_model_path).exists():
            QMessageBox.warning(
                self,
                "Model Not Found",
                "Seçili experiment için model bulunamadı."
            )
            return

        self.current_analysis_type = "original"
        self._start_prediction(
            model_path=self.analysis_model_path,
            image_path=self.image_path,
            use_preprocessing=False
        )

    def _analyze_preprocessed_image(self):
        if not self.image_path:
            QMessageBox.warning(
                self,
                "Image Not Found",
                "Önce Görsel Seçiniz butonu ile bir el yazısı görseli seçmelisin."
            )
            return

        if not self.analysis_model_path or not Path(self.analysis_model_path).exists():
            QMessageBox.warning(
                self,
                "Model Not Found",
                "Seçili experiment için model bulunamadı."
            )
            return

        try:
            processed_path = self.controller.preprocess_image(self.image_path)
            self.processed_image_path = processed_path
            self._update_processed_image(processed_path)

            self.current_analysis_type = "enhanced"
            self._start_prediction(
                model_path=self.analysis_model_path,
                image_path=processed_path,
                use_preprocessing=False
            )

        except Exception as error:
            QMessageBox.critical(
                self,
                "Preprocessing Error",
                str(error)
            )

    def _start_prediction(self, model_path: str, image_path: str, use_preprocessing: bool):
        if not image_path:
            return

        if self.worker is not None and self.worker.isRunning():
            return

        self._set_controls_enabled(False)

        self.worker = AnalysisWorker(
            controller=self.controller,
            model_path=model_path,
            image_path=image_path,
            use_preprocessing=use_preprocessing,
            task_type=self.current_task_type
        )

        self.worker.result_ready.connect(self._on_prediction_finished)
        self.worker.error_ready.connect(self._on_prediction_failed)

        self.worker.finished.connect(self.worker.deleteLater)
        self.worker.finished.connect(self._cleanup_worker)

        self.worker.start()

    def _cleanup_worker(self):
        self.worker = None

    def _on_prediction_finished(self, result: Dict[str, Any]):
        self._update_prediction_panel(result)
        self._update_sample_result(result)
        self._store_session_prediction_result(result)
        self._set_controls_enabled(True)

    def _on_prediction_failed(self, error_message: str):
        QMessageBox.critical(self, "Prediction Error", error_message)
        self._set_controls_enabled(True)

    def closeEvent(self, event):
        if self.worker is not None and self.worker.isRunning():
            self.worker.quit()
            self.worker.wait(3000)

        event.accept()

    def _update_processed_image(self, image_path: str):
        pixmap = self._make_pixmap_for_label(
            image_path=image_path,
            label=self.processed_image_label
        )

        self.processed_image_label.setPixmap(pixmap)

    def _extract_trait_values_from_result(self, result: Dict[str, Any]) -> Dict[str, float]:
        possible_dict_keys = [
            "trait_scores",
            "predicted_scores",
            "predictions",
            "scores",
            "traits"
        ]

        for key in possible_dict_keys:
            value = result.get(key)

            if isinstance(value, dict):
                return {
                    trait: float(value.get(trait, 0.0))
                    for trait in OCEAN_ORDER
                }

        values = result.get("traits")

        if values is None:
            values = result.get("prediction")

        if values is None:
            values = result.get("predicted_values")

        if isinstance(values, list):
            class_names = result.get("class_names", MODEL_LABEL_ORDER)

            trait_map = {}

            for name, value in zip(class_names, values):
                try:
                    trait_map[name] = float(value)
                except Exception:
                    trait_map[name] = 0.0

            return {
                trait: trait_map.get(trait, 0.0)
                for trait in OCEAN_ORDER
            }

        return {
            trait: 0.0
            for trait in OCEAN_ORDER
        }

    def _update_prediction_panel(self, result: Dict[str, Any]):
        trait_values = self._extract_trait_values_from_result(result)

        for trait in OCEAN_ORDER:
            raw_value = trait_values.get(trait, 0.0)

            if self.current_task_type == "regression":
                display_value = raw_value

                if 0.0 <= raw_value <= 5.0:
                    bar_value = raw_value * 20.0
                    label_text = f"{raw_value:.2f}/5"
                else:
                    bar_value = raw_value
                    label_text = f"{raw_value:.1f}"
            else:
                bar_value = raw_value
                label_text = f"{raw_value:.1f}%"

            bar_value = max(0.0, min(100.0, bar_value))

            self.trait_bars[trait].setValue(int(bar_value))
            self.trait_value_labels[trait].setText(label_text)

    def _update_sample_result(self, result: Dict[str, Any]):
        if self.current_task_type != "classification":
            self.sample_result_label.setText("Regression çıktısı: sürekli değer tahmini yapıldı.")
            return

        if not self.current_expected_class:
            return

        predicted_class = result.get("predicted_class", "—")
        is_correct = predicted_class == self.current_expected_class

        self.sample_result_label.setText(
            f"Son Örneğin Sonucu: Beklenen={self.current_expected_class} | "
            f"Tahmin={predicted_class} | {'Doğru' if is_correct else 'Yanlış'}"
        )

    def _load_sample_dataset(self, dataset_id: str):
        root = self.controller.get_dataset_sample_root(dataset_id)
        extensions = {".png", ".jpg", ".jpeg", ".bmp"}

        self.demo_image_paths = {}
        self.demo_indices = {}

        for class_name in MODEL_LABEL_ORDER:
            self.demo_image_paths[class_name] = []
            self.demo_indices[class_name] = 0

            if not root:
                continue

            class_dir = root / class_name

            if not class_dir.exists():
                continue

            images = [
                path for path in class_dir.rglob("*")
                if path.suffix.lower() in extensions
            ]

            self.demo_image_paths[class_name] = sorted(images)

    def _load_next_sample(self, class_name: str):
        if self.current_task_type != "classification":
            QMessageBox.information(
                self,
                "Regression Deneyi",
                "Regression deneylerinde sınıf klasörlerinden örnek seçimi kullanılmaz."
            )
            return

        image_paths = self.demo_image_paths.get(class_name, [])

        if not image_paths:
            QMessageBox.warning(
                self,
                "Sample Not Found",
                f"{class_name} için test görseli bulunamadı."
            )
            return

        index = self.demo_indices[class_name]

        if index >= len(image_paths):
            index = 0
            self.demo_indices[class_name] = 0

        selected_path = image_paths[index]
        self.demo_indices[class_name] += 1

        self.current_expected_class = class_name

        prepared_path = self.controller.prepare_gui_image(str(selected_path))
        self._load_image_to_gui(
            prepared_path,
            display_name=f"{class_name}/{selected_path.name}",
            image_key=str(selected_path.resolve())
        )
        self._analyze_original_image()

    def _load_random_test_sample(self):
        if self.current_task_type != "classification":
            QMessageBox.information(
                self,
                "Regression Deneyi",
                "Regression deneylerinde rastgele sınıf örneği kullanılmaz. Manuel görsel seçebilirsiniz."
            )
            return

        all_images = []

        for class_name, image_paths in self.demo_image_paths.items():
            for path in image_paths:
                all_images.append((class_name, path))

        if not all_images:
            QMessageBox.warning(
                self,
                "Sample Not Found",
                "Seçili dataset için test görseli bulunamadı."
            )
            return

        class_name, selected_path = random.choice(all_images)

        self.current_expected_class = class_name

        prepared_path = self.controller.prepare_gui_image(str(selected_path))
        self._load_image_to_gui(
            prepared_path,
            display_name=f"{class_name}/{selected_path.name}",
            image_key=str(selected_path.resolve())
        )
        self._analyze_original_image()

    def _update_active_model_text(self):
        if not self.analysis_experiment or not self.analysis_checkpoint:
            self.active_model_label.setText("Seçili Model: —")
            self.active_model_metric_label.setText("Performans: —")
            return

        task_type = self.analysis_experiment.get("task", "classification")

        self.active_model_label.setText(
            f"Seçili Model: {self.analysis_experiment['experiment_id']} / "
            f"{self.analysis_checkpoint['name']} / {task_type.upper()}"
        )

        metrics = self.controller.get_best_metrics_for_checkpoint(
            self.analysis_checkpoint
        )

        if not metrics:
            self.active_model_metric_label.setText("Performans: metrics not found")
            return

        if task_type == "regression":
            rmse = self._metric_value(metrics, ["rmse", "test_rmse", "avg_rmse", "RMSE"])
            mae = self._metric_value(metrics, ["mae", "test_mae", "avg_mae", "MAE"])
            pcc = self._metric_value(metrics, ["pcc", "pearson", "avg_pcc", "PCC"])

            self.active_model_metric_label.setText(
                f"Performans: RMSE={self._format_metric(rmse)} | "
                f"MAE={self._format_metric(mae)} | "
                f"PCC={self._format_metric(pcc)}"
            )
            return

        accuracy = metrics.get("accuracy", "—")
        macro_f1 = metrics.get("macro_f1", "—")

        self.active_model_metric_label.setText(
            f"Performans: Acc={accuracy} | Macro F1={macro_f1}"
        )

    def _set_controls_enabled(self, enabled: bool):
        self.experiment_combo.setEnabled(enabled)
        self.browse_button.setEnabled(enabled)
        self.analyze_original_button.setEnabled(enabled)
        self.preprocess_button.setEnabled(enabled)
        self.open_comparison_button.setEnabled(enabled)
        self.open_prediction_comparison_button.setEnabled(enabled)

        for button in self.sample_buttons.values():
            button.setEnabled(enabled)


def main():
    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLE)

    window = HandwritingDashboard()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
