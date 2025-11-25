#!/usr/bin/env python3
"""
Mouse Sperm Head Detection Workbench - GUI Application
Usage: python gui.py
"""

import sys
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import tifffile
import imageio
from dataclasses import dataclass
from core import detect_sperm_coordinates_enhanced


from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QWidget,
    QVBoxLayout,
    QHBoxLayout,
    QPushButton,
    QSlider,
    QSpinBox,
    QDoubleSpinBox,
    QLabel,
    QGroupBox,
    QComboBox,
    QCheckBox,
    QLineEdit,
    QFileDialog,
    QStatusBar,
    QGraphicsView,
    QGraphicsScene,
    QTableWidget,
    QTableWidgetItem,
    QSplitter,
    QScrollArea,
    QToolButton,
    QProgressBar,
    QMessageBox,
    QStyle,
)
from PyQt6.QtCore import (
    Qt,
    QObject,
    pyqtSignal,
    QTimer,
    QRectF,
    QThreadPool,
    QRunnable,
    QEvent,
)
from PyQt6.QtGui import (
    QImage,
    QPixmap,
    QAction,
    QKeySequence,
    QPainter,
)


@dataclass
class DetectionParams:
    """Parameter container matching command-line args"""

    min_area: int = 20
    max_area: int = 45
    min_aspect: float = 1.2
    max_aspect: float = 3.0
    min_solidity: float = 0.65
    threshold: int = 10
    blur_radius: float = 0.5
    marker_color: str = "red"
    marker_size: int = 3
    marker_shape: str = "circle"


# DetectionWorker becomes much simpler
class DetectionWorker(QRunnable):
    """Worker for running detection in thread pool"""

    def __init__(self, frame_data: np.ndarray, params: DetectionParams, frame_idx: int):
        super().__init__()
        self.frame_data = frame_data
        self.params = params
        self.frame_idx = frame_idx
        self.signals = WorkerSignals()

    def run(self):
        """Run detection algorithm"""
        try:
            self.signals.started.emit()
            detections, overlay = self._run_detection(self.frame_data, self.params)
            self.signals.finished.emit(
                overlay, {"frame_idx": self.frame_idx, "detections": detections}
            )
        except Exception as e:
            self.signals.error.emit(str(e))

    def _run_detection(
        self, image: np.ndarray, params: DetectionParams
    ) -> Tuple[List[Dict], np.ndarray]:
        """Use enhanced detection function with detailed metrics"""
        # Convert to grayscale if needed
        if image.ndim == 3:
            if image.shape[2] == 3:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            elif image.shape[2] == 4:
                image_gray = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
            else:
                image_gray = image[..., 0]  # Take first channel
        else:
            image_gray = image

        # Use enhanced detection function to get detailed metrics
        enhanced_detections = detect_sperm_coordinates_enhanced(
            image_gray,
            min_area=params.min_area,
            max_area=params.max_area,
            min_aspect=params.min_aspect,
            max_aspect=params.max_aspect,
            min_solidity=params.min_solidity,
            threshold=params.threshold,
            blur_radius=params.blur_radius,
        )

        # Convert to detection list format with actual metrics
        detections = []
        for idx, detection in enumerate(enhanced_detections):
            detections.append(
                {
                    "id": idx,
                    "x": int(detection["x"]),
                    "y": int(detection["y"]),
                    "area": detection["area"],
                    "aspect": detection["aspect_ratio"],
                    "solidity": detection["solidity"],
                    # Include all detection parameters for GUI export
                    "threshold": detection["threshold"],
                    "blur_radius": detection["blur_radius"],
                    "min_area": detection["min_area"],
                    "max_area": detection["max_area"],
                    "min_aspect": detection["min_aspect"],
                    "max_aspect": detection["max_aspect"],
                    "min_solidity": detection["min_solidity"],
                }
            )

        # Create overlay using coordinates only
        coordinates = [(det["x"], det["y"]) for det in enhanced_detections]
        marked_frame = self._add_markers_to_frame(
            image_gray,
            coordinates,
            params.marker_color,
            params.marker_size,
            params.marker_shape,
        )

        return detections, marked_frame

    def _add_markers_to_frame(
        self,
        frame: np.ndarray,
        coordinates: List[Tuple[float, float]],
        color: str,
        size: int,
        shape: str,
    ) -> np.ndarray:
        """Add markers to frame (simplified version)"""
        if len(frame.shape) == 2:
            marked_frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
        else:
            marked_frame = frame.copy()

        colors = {
            "red": (0, 0, 255),
            "blue": (255, 0, 0),
            "green": (0, 255, 0),
            "yellow": (0, 255, 255),
            "white": (255, 255, 255),
            "cyan": (255, 255, 0),
            "magenta": (255, 0, 255),
        }
        marker_color = colors.get(color.lower(), (0, 0, 255))

        for x, y in coordinates:
            cv2.circle(marked_frame, (int(x), int(y)), size, marker_color, -1)

        return marked_frame


class WorkerSignals(QObject):
    """Signal holder for QRunnable worker"""

    started = pyqtSignal()
    finished = pyqtSignal(np.ndarray, dict)
    error = pyqtSignal(str)


class TIFFLoader:
    """Helper class for loading and managing TIFF stacks"""

    def __init__(self):
        self.processed_path: Optional[Path] = None
        self.overlay_path: Optional[Path] = None
        self.processed_data: Optional[np.ndarray] = None
        self.overlay_data: Optional[np.ndarray] = None
        self.current_frame: int = 0

    def load_processed(self, path: Path) -> bool:
        """Load processed TIFF file"""
        try:
            self.processed_path = path
            self.processed_data = tifffile.imread(path)
            # Ensure 3D array (frames, height, width)
            if self.processed_data.ndim == 2:
                self.processed_data = self.processed_data[np.newaxis, ...]
            return True
        except Exception as e:
            print(f"Error loading processed TIFF: {e}")
            return False

    def load_overlay(self, path: Path) -> bool:
        """Load unprocessed movie for overlay"""
        try:
            self.overlay_path = path
            self.overlay_data = tifffile.imread(path)
            if self.overlay_data.ndim == 2:
                self.overlay_data = self.overlay_data[np.newaxis, ...]
            return True
        except Exception as e:
            print(f"Error loading overlay TIFF: {e}")
            return False

    def get_frame(self, frame_idx: int, overlay: bool = False) -> Optional[np.ndarray]:
        """Get specific frame"""
        data = self.overlay_data if overlay else self.processed_data
        if data is None or frame_idx >= data.shape[0]:
            return None
        return data[frame_idx].copy()

    @property
    def num_frames(self) -> int:
        return self.processed_data.shape[0] if self.processed_data is not None else 0


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Sperm Head Detection Workbench")
        self.setMinimumSize(1400, 900)

        self.tiff_loader = TIFFLoader()
        self.params = DetectionParams()
        self.detection_cache: Dict[int, Tuple[List[Dict], np.ndarray]] = {}
        self.is_processing = False

        self.thread_pool = QThreadPool()
        self.thread_pool.setMaxThreadCount(4)

        self._setup_ui()
        self._connect_signals()
        self._update_ui_state()

        # Playback timer
        self.playback_timer = QTimer()
        self.playback_timer.timeout.connect(self._next_frame)
        self.playback_fps = 5

        # Debounce timer for parameter changes
        self.param_timer = QTimer()
        self.param_timer.setSingleShot(True)
        self.param_timer.timeout.connect(self._on_parameter_changed_complete)

    def _setup_ui(self):
        """Setup the user interface"""
        # Create main splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left panel
        self.left_panel = self._create_left_panel()
        splitter.addWidget(self.left_panel)

        # Middle panel
        self.middle_panel = self._create_middle_panel()
        splitter.addWidget(self.middle_panel)

        # Right panel
        self.right_panel = self._create_right_panel()
        splitter.addWidget(self.right_panel)

        # Set splitter proportions
        splitter.setSizes([320, 800, 280])

        self.setCentralWidget(splitter)
        self._create_status_bar()
        self._create_menu_bar()

    def _create_left_panel(self) -> QScrollArea:
        """Create left control panel with scrollable area"""
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(300)
        scroll.setMaximumWidth(400)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(15)

        # File Management
        file_group = QGroupBox("File Management")
        file_layout = QVBoxLayout()

        # Input file
        input_layout = QHBoxLayout()
        self.input_lineedit = QLineEdit()
        self.input_button = QPushButton("Browse...")
        input_layout.addWidget(QLabel("Input TIF:"))
        input_layout.addWidget(self.input_lineedit)
        input_layout.addWidget(self.input_button)
        file_layout.addLayout(input_layout)

        # Output directory
        output_layout = QHBoxLayout()
        self.output_lineedit = QLineEdit()
        self.output_button = QPushButton("Browse...")
        output_layout.addWidget(QLabel("Output Dir:"))
        output_layout.addWidget(self.output_lineedit)
        output_layout.addWidget(self.output_button)
        file_layout.addLayout(output_layout)

        # Overlay movie
        overlay_layout = QHBoxLayout()
        self.overlay_checkbox = QCheckBox("Overlay Movie:")
        self.overlay_lineedit = QLineEdit()
        self.overlay_lineedit.setEnabled(False)
        self.overlay_button = QPushButton("Browse...")
        self.overlay_button.setEnabled(False)
        overlay_layout.addWidget(self.overlay_checkbox)
        overlay_layout.addWidget(self.overlay_lineedit)
        overlay_layout.addWidget(self.overlay_button)
        file_layout.addLayout(overlay_layout)

        # Load status
        self.load_status_label = QLabel("Status: No file loaded")
        self.load_status_label.setStyleSheet("color: #d73a49; font-weight: bold;")
        file_layout.addWidget(self.load_status_label)

        file_group.setLayout(file_layout)
        layout.addWidget(file_group)

        # Parameter controls
        param_group = QGroupBox("Detection Parameters (Live Tuning)")
        param_layout = QVBoxLayout()

        self.param_widgets = {}

        # Create slider+spinbox pairs
        for name, min_val, max_val, default, step, is_float in [
            ("threshold", 0, 255, 10, 1, False),
            ("blur_radius", 0.0, 10.0, 0.5, 0.1, True),
            ("min_area", 5, 100, 20, 1, False),
            ("max_area", 20, 200, 45, 1, False),
            ("min_aspect", 1.0, 5.0, 1.2, 0.1, True),
            ("max_aspect", 1.0, 10.0, 3.0, 0.1, True),
            ("min_solidity", 0.0, 1.0, 0.65, 0.01, True),
        ]:
            row = QHBoxLayout()
            row.addWidget(QLabel(f"{name.replace('_', ' ').title()}:"))

            if is_float:
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(int(min_val / step), int(max_val / step))
                slider.setValue(int(default / step))
                spinbox = QDoubleSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setSingleStep(step)
                spinbox.setValue(default)
                spinbox.setDecimals(2)
                self.param_widgets[name] = (slider, spinbox)

                slider.valueChanged.connect(
                    lambda val, s=spinbox, st=step: s.setValue(val * st)
                )
                spinbox.valueChanged.connect(
                    lambda val, sl=slider, st=step: sl.setValue(int(val / st))
                )
            else:
                slider = QSlider(Qt.Orientation.Horizontal)
                slider.setRange(min_val, max_val)
                slider.setValue(default)
                spinbox = QSpinBox()
                spinbox.setRange(min_val, max_val)
                spinbox.setValue(default)
                self.param_widgets[name] = (slider, spinbox)

                slider.valueChanged.connect(spinbox.setValue)
                spinbox.valueChanged.connect(slider.setValue)

            # Connect to parameter changed signal
            spinbox.valueChanged.connect(self._on_parameter_changed)

            row.addWidget(slider)
            row.addWidget(spinbox)
            param_layout.addLayout(row)

        param_group.setLayout(param_layout)
        layout.addWidget(param_group)

        # Visualization controls
        viz_group = QGroupBox("Visualization")
        viz_layout = QVBoxLayout()

        # Marker color
        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Marker Color:"))
        self.marker_color_combo = QComboBox()
        self.marker_color_combo.addItems(
            ["red", "blue", "green", "yellow", "white", "cyan", "magenta"]
        )
        self.marker_color_combo.setCurrentText("red")
        color_layout.addWidget(self.marker_color_combo)
        viz_layout.addLayout(color_layout)

        # Marker shape
        shape_layout = QHBoxLayout()
        shape_layout.addWidget(QLabel("Marker Shape:"))
        self.marker_shape_combo = QComboBox()
        self.marker_shape_combo.addItems(
            ["circle", "cross", "square", "plus", "diamond"]
        )
        shape_layout.addWidget(self.marker_shape_combo)
        viz_layout.addLayout(shape_layout)

        # Marker size
        size_layout = QHBoxLayout()
        size_layout.addWidget(QLabel("Marker Size:"))
        self.marker_size_slider = QSlider(Qt.Orientation.Horizontal)
        self.marker_size_slider.setRange(1, 10)
        self.marker_size_slider.setValue(3)
        self.marker_size_label = QLabel("3")
        size_layout.addWidget(self.marker_size_slider)
        size_layout.addWidget(self.marker_size_label)
        self.marker_size_slider.valueChanged.connect(
            lambda v: self.marker_size_label.setText(str(v))
        )
        viz_layout.addLayout(size_layout)

        # Overlay opacity
        opacity_layout = QHBoxLayout()
        opacity_layout.addWidget(QLabel("Overlay Opacity:"))
        self.opacity_slider = QSlider(Qt.Orientation.Horizontal)
        self.opacity_slider.setRange(0, 100)
        self.opacity_slider.setValue(75)
        self.opacity_label = QLabel("75%")
        opacity_layout.addWidget(self.opacity_slider)
        opacity_layout.addWidget(self.opacity_label)
        self.opacity_slider.valueChanged.connect(
            lambda v: self.opacity_label.setText(f"{v}%")
        )
        viz_layout.addLayout(opacity_layout)

        viz_group.setLayout(viz_layout)
        layout.addWidget(viz_group)

        # Detection buttons
        action_group = QGroupBox("Actions")
        action_layout = QVBoxLayout()

        button_layout = QHBoxLayout()
        self.detect_current_button = QPushButton("▶ Detect Current")
        self.process_stack_button = QPushButton("▶▶ Process Stack")
        button_layout.addWidget(self.detect_current_button)
        button_layout.addWidget(self.process_stack_button)
        action_layout.addLayout(button_layout)

        self.cancel_button = QPushButton("⏸ Cancel")
        self.cancel_button.setEnabled(False)
        action_layout.addWidget(self.cancel_button)

        action_group.setLayout(action_layout)
        layout.addWidget(action_group)

        # Export options
        export_group = QGroupBox("Export")
        export_layout = QVBoxLayout()

        export_button_layout = QHBoxLayout()
        self.save_params_button = QPushButton("Save Params")
        self.save_snapshot_button = QPushButton("Save Snapshot")
        export_button_layout.addWidget(self.save_params_button)
        export_button_layout.addWidget(self.save_snapshot_button)
        export_layout.addLayout(export_button_layout)

        export_button_layout2 = QHBoxLayout()
        self.export_csv_button = QPushButton("Export CSV")
        self.export_video_button = QPushButton("Export Video")
        export_button_layout2.addWidget(self.export_csv_button)
        export_button_layout2.addWidget(self.export_video_button)
        export_layout.addLayout(export_button_layout2)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        layout.addStretch()

        scroll.setWidget(container)
        return scroll

    def _create_middle_panel(self) -> QWidget:
        """Create middle display panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)

        # Graphics view for image display
        self.graphics_view = QGraphicsView()
        self.graphics_view.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.graphics_view.setOptimizationFlag(
            QGraphicsView.OptimizationFlag.DontAdjustForAntialiasing
        )
        self.graphics_scene = QGraphicsScene()
        self.graphics_view.setScene(self.graphics_scene)
        layout.addWidget(self.graphics_view)

        # Navigation toolbar
        nav_widget = QWidget()
        nav_layout = QVBoxLayout(nav_widget)
        nav_layout.setContentsMargins(10, 5, 10, 5)

        # Frame slider with controls row
        slider_controls_layout = QHBoxLayout()

        # Navigation buttons
        self.first_frame_button = QToolButton()
        self.first_frame_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipBackward)
        )
        self.prev_frame_button = QToolButton()
        self.prev_frame_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekBackward)
        )
        self.play_button = QToolButton()
        self.play_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
        )
        self.next_frame_button = QToolButton()
        self.next_frame_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSeekForward)
        )
        self.last_frame_button = QToolButton()
        self.last_frame_button.setIcon(
            self.style().standardIcon(QStyle.StandardPixmap.SP_MediaSkipForward)
        )

        slider_controls_layout.addWidget(self.first_frame_button)
        slider_controls_layout.addWidget(self.prev_frame_button)
        slider_controls_layout.addWidget(self.play_button)
        slider_controls_layout.addWidget(self.next_frame_button)
        slider_controls_layout.addWidget(self.last_frame_button)

        # Frame indicator
        slider_controls_layout.addStretch()
        slider_controls_layout.addWidget(QLabel("Frame:"))
        self.frame_spinbox = QSpinBox()
        self.frame_spinbox.setEnabled(False)
        slider_controls_layout.addWidget(self.frame_spinbox)
        self.frame_total_label = QLabel("/ 0")
        slider_controls_layout.addWidget(self.frame_total_label)

        # FPS control
        slider_controls_layout.addStretch()
        slider_controls_layout.addWidget(QLabel("FPS:"))
        self.fps_combo = QComboBox()
        self.fps_combo.addItems(["1", "5", "10", "15", "30"])
        self.fps_combo.setCurrentText("5")
        self.fps_combo.setFixedWidth(60)
        slider_controls_layout.addWidget(self.fps_combo)

        nav_layout.addLayout(slider_controls_layout)

        # Frame slider
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setTickPosition(QSlider.TickPosition.TicksBelow)
        self.frame_slider.setTickInterval(1)
        nav_layout.addWidget(self.frame_slider)

        # Bottom controls row
        bottom_controls_layout = QHBoxLayout()

        # Overlay toggle
        bottom_controls_layout.addWidget(QLabel("Overlay:"))
        self.overlay_toggle = QCheckBox("On")
        self.overlay_toggle.setChecked(True)
        bottom_controls_layout.addWidget(self.overlay_toggle)
        bottom_controls_layout.addStretch()

        # Display mode
        bottom_controls_layout.addWidget(QLabel("Mode:"))
        self.mode_combo = QComboBox()
        self.mode_combo.addItems(
            ["Processed", "Unprocessed", "Side-by-Side", "Difference"]
        )
        bottom_controls_layout.addWidget(self.mode_combo)

        nav_layout.addLayout(bottom_controls_layout)

        layout.addWidget(nav_widget)
        return container

    def _create_right_panel(self) -> QWidget:
        """Create right statistics panel"""
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Current frame stats
        self.frame_stats_group = QGroupBox("Current Frame Stats")
        stats_layout = QVBoxLayout()

        self.frame_info_label = QLabel("Frame: -")
        self.detection_count_label = QLabel("Detections: 0")
        self.avg_area_label = QLabel("Avg Area: -")
        self.avg_aspect_label = QLabel("Avg Aspect: -")
        self.avg_solidity_label = QLabel("Avg Solidity: -")

        stats_layout.addWidget(self.frame_info_label)
        stats_layout.addWidget(self.detection_count_label)
        stats_layout.addWidget(self.avg_area_label)
        stats_layout.addWidget(self.avg_aspect_label)
        stats_layout.addWidget(self.avg_solidity_label)

        self.frame_stats_group.setLayout(stats_layout)
        layout.addWidget(self.frame_stats_group)

        # Detection log table
        log_group = QGroupBox("Detection Log")
        log_layout = QVBoxLayout()

        self.log_table = QTableWidget()
        self.log_table.setColumnCount(6)
        self.log_table.setHorizontalHeaderLabels(
            ["ID", "Area", "Aspect", "Solidity", "X", "Y"]
        )
        self.log_table.horizontalHeader().setStretchLastSection(True)
        self.log_table.setAlternatingRowColors(True)
        self.log_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        log_layout.addWidget(self.log_table)

        log_group.setLayout(log_layout)
        layout.addWidget(log_group)

        # Stack summary
        self.summary_group = QGroupBox("Stack Summary")
        summary_layout = QVBoxLayout()

        self.total_frames_label = QLabel("Total Frames: 0")
        self.frames_with_det_label = QLabel("Frames w/ Detections: 0 (0%)")
        self.total_dets_label = QLabel("Total Detections: 0")
        self.avg_per_frame_label = QLabel("Average/Frame: 0")
        self.processing_time_label = QLabel("Processing Time: -")

        summary_layout.addWidget(self.total_frames_label)
        summary_layout.addWidget(self.frames_with_det_label)
        summary_layout.addWidget(self.total_dets_label)
        summary_layout.addWidget(self.avg_per_frame_label)
        summary_layout.addWidget(self.processing_time_label)

        self.summary_group.setLayout(summary_layout)
        layout.addWidget(self.summary_group)

        # Filters
        filter_group = QGroupBox("Filters")
        filter_layout = QVBoxLayout()

        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("Show Frames:"))
        self.filter_combo = QComboBox()
        self.filter_combo.addItems(["All", "With Detections", "Without Detections"])
        filter_row.addWidget(self.filter_combo)
        filter_layout.addLayout(filter_row)

        min_det_layout = QHBoxLayout()
        min_det_layout.addWidget(QLabel("Min Detections:"))
        self.min_det_spinbox = QSpinBox()
        self.min_det_spinbox.setRange(0, 100)
        min_det_layout.addWidget(self.min_det_spinbox)
        filter_layout.addLayout(min_det_layout)

        filter_button_layout = QHBoxLayout()
        self.apply_filter_button = QPushButton("Apply Filter")
        self.reset_filter_button = QPushButton("Reset")
        filter_button_layout.addWidget(self.apply_filter_button)
        filter_button_layout.addWidget(self.reset_filter_button)
        filter_layout.addLayout(filter_button_layout)

        filter_group.setLayout(filter_layout)
        layout.addWidget(filter_group)

        layout.addStretch()
        return container

    def _create_status_bar(self):
        """Create status bar"""
        self.status_bar = QStatusBar()
        self.status_label = QLabel("Ready")
        self.status_bar.addWidget(self.status_label)

        self.progress_bar = QProgressBar()
        self.progress_bar.setVisible(False)
        self.progress_bar.setFixedWidth(200)
        self.status_bar.addPermanentWidget(self.progress_bar)

        self.tuning_indicator = QLabel("⚡ Live Tuning: Inactive")
        self.status_bar.addPermanentWidget(self.tuning_indicator)

        self.setStatusBar(self.status_bar)

    def _create_menu_bar(self):
        """Create menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("&File")

        self.load_action = QAction("&Load Processed TIF...", self)
        self.load_action.setShortcut(QKeySequence("Ctrl+O"))
        file_menu.addAction(self.load_action)

        self.load_overlay_action = QAction("Load &Overlay Movie...", self)
        self.load_overlay_action.setShortcut(QKeySequence("Ctrl+Shift+O"))
        file_menu.addAction(self.load_overlay_action)

        file_menu.addSeparator()

        self.exit_action = QAction("E&xit", self)
        self.exit_action.setShortcut(QKeySequence("Ctrl+Q"))
        self.exit_action.triggered.connect(self.close)
        file_menu.addAction(self.exit_action)

        # View menu
        view_menu = menubar.addMenu("&View")

        self.reset_zoom_action = QAction("&Reset Zoom", self)
        self.reset_zoom_action.setShortcut(QKeySequence("Ctrl+0"))
        view_menu.addAction(self.reset_zoom_action)

        # Help menu
        help_menu = menubar.addMenu("&Help")

        self.about_action = QAction("&About", self)
        self.about_action.triggered.connect(self._show_about)
        help_menu.addAction(self.about_action)

    def _connect_signals(self):
        """Connect all UI signals to slots"""
        # File loading
        self.input_button.clicked.connect(lambda: self._load_file("processed"))
        self.output_button.clicked.connect(self._select_output_dir)
        self.overlay_checkbox.toggled.connect(self._toggle_overlay_ui)
        self.overlay_button.clicked.connect(lambda: self._load_file("overlay"))

        # Navigation
        self.first_frame_button.clicked.connect(lambda: self._set_frame(0))
        self.prev_frame_button.clicked.connect(
            lambda: self._set_frame(max(0, self.tiff_loader.current_frame - 1))
        )
        self.next_frame_button.clicked.connect(
            lambda: self._set_frame(
                min(self.tiff_loader.num_frames - 1, self.tiff_loader.current_frame + 1)
            )
        )
        self.last_frame_button.clicked.connect(
            lambda: self._set_frame(self.tiff_loader.num_frames - 1)
        )
        self.play_button.clicked.connect(self._toggle_playback)
        self.frame_spinbox.valueChanged.connect(self._set_frame)
        self.frame_slider.valueChanged.connect(self._set_frame)
        self.fps_combo.currentTextChanged.connect(
            lambda text: setattr(self, "playback_fps", int(text))
        )

        # Display controls
        self.overlay_toggle.toggled.connect(self._update_display)

        # Use activated with delay to ensure dropdown closes
        self.mode_combo.activated.connect(
            lambda: QTimer.singleShot(50, self._update_display)
        )

        # Parameters
        for slider, spinbox in self.param_widgets.values():
            spinbox.valueChanged.connect(self._on_parameter_changed)

        # Visualization dropdowns - use activated with delay
        self.marker_color_combo.activated.connect(
            lambda: QTimer.singleShot(50, self._on_parameter_changed)
        )
        self.marker_shape_combo.activated.connect(
            lambda: QTimer.singleShot(50, self._on_parameter_changed)
        )
        self.marker_size_slider.valueChanged.connect(self._on_parameter_changed)
        self.opacity_slider.valueChanged.connect(self._update_display)

        # Actions
        self.detect_current_button.clicked.connect(self._detect_current_frame)
        self.process_stack_button.clicked.connect(self._process_stack)
        self.cancel_button.clicked.connect(self._cancel_processing)

        # Export
        self.save_params_button.clicked.connect(self._save_params)
        self.save_snapshot_button.clicked.connect(self._save_snapshot)
        self.export_csv_button.clicked.connect(self._export_csv)
        self.export_video_button.clicked.connect(self._export_video)

        # Filters
        self.apply_filter_button.clicked.connect(self._apply_filter)
        self.reset_filter_button.clicked.connect(self._reset_filter)

        # Menu actions
        self.load_action.triggered.connect(lambda: self._load_file("processed"))
        self.load_overlay_action.triggered.connect(lambda: self._load_file("overlay"))
        self.reset_zoom_action.triggered.connect(self._reset_zoom)

    def _load_file(self, file_type: str):
        """Load TIFF file"""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            f"Select {'Processed' if file_type == 'processed' else 'Overlay'} TIFF",
            "",
            "TIFF Files (*.tif *.tiff)",
        )

        if file_path:
            path = Path(file_path)
            if file_type == "processed":
                self.input_lineedit.setText(str(path))
                if self.tiff_loader.load_processed(path):
                    self._on_file_loaded()
            else:
                self.overlay_lineedit.setText(str(path))
                if self.tiff_loader.load_overlay(path):
                    self.overlay_checkbox.setChecked(True)
                    self._update_display()

            self._update_ui_state()

    def _select_output_dir(self):
        """Select output directory"""
        dir_path = QFileDialog.getExistingDirectory(self, "Select Output Directory")
        if dir_path:
            self.output_lineedit.setText(dir_path)

    def _toggle_overlay_ui(self, checked: bool):
        """Enable/disable overlay controls"""
        self.overlay_lineedit.setEnabled(checked)
        self.overlay_button.setEnabled(checked)
        self._update_display()

    def _on_file_loaded(self):
        """Handle successful file load"""
        self.load_status_label.setText(
            f"✓ Loaded ({self.tiff_loader.num_frames} frames)"
        )
        self.load_status_label.setStyleSheet("color: #28a745; font-weight: bold;")

        self.frame_spinbox.setEnabled(True)
        self.frame_spinbox.setRange(0, self.tiff_loader.num_frames - 1)
        self.frame_total_label.setText(f"/ {self.tiff_loader.num_frames - 1}")

        # Configure frame slider
        self.frame_slider.setEnabled(True)
        self.frame_slider.setRange(0, self.tiff_loader.num_frames - 1)
        # Set tick interval to show reasonable number of ticks
        tick_interval = max(1, self.tiff_loader.num_frames // 20)
        self.frame_slider.setTickInterval(tick_interval)

        self._set_frame(0)
        self._update_summary()

    def _set_frame(self, frame_idx: int):
        """Set current frame and update display"""
        if self.tiff_loader.num_frames == 0:
            return

        # Ensure frame_idx is within valid range
        frame_idx = max(0, min(frame_idx, self.tiff_loader.num_frames - 1))

        self.tiff_loader.current_frame = frame_idx

        # Update spinbox without triggering signals
        self.frame_spinbox.blockSignals(True)
        self.frame_spinbox.setValue(frame_idx)
        self.frame_spinbox.blockSignals(False)

        # Update slider without triggering signals
        self.frame_slider.blockSignals(True)
        self.frame_slider.setValue(frame_idx)
        self.frame_slider.blockSignals(False)

        self._update_display()
        self._update_frame_stats()

    def _update_display(self):
        """Update the main display"""
        if self.tiff_loader.num_frames == 0:
            return

        mode = self.mode_combo.currentText()
        show_overlay = self.overlay_toggle.isChecked()

        # Clear scene
        self.graphics_scene.clear()

        if mode == "Side-by-Side":
            self._display_side_by_side()
        elif mode == "Difference":
            self._display_difference()
        else:
            self._display_single(mode, show_overlay)

        self.status_label.setText(
            f"Frame {self.tiff_loader.current_frame}/{self.tiff_loader.num_frames - 1}"
        )

    def _display_single(self, mode: str, show_overlay: bool):
        """Display single image with optional overlay"""
        # Get base image
        use_overlay = (
            mode == "Unprocessed" and self.tiff_loader.overlay_data is not None
        )
        frame = self.tiff_loader.get_frame(self.tiff_loader.current_frame, use_overlay)

        if frame is None:
            return

        # Get detection overlay if available
        if show_overlay and self.tiff_loader.current_frame in self.detection_cache:
            _, overlay = self.detection_cache[self.tiff_loader.current_frame]
            if overlay is not None:
                # Blend with opacity
                opacity = self.opacity_slider.value() / 100.0
                if frame.ndim == 2:
                    frame_rgb = np.stack([frame, frame, frame], axis=-1)
                else:
                    frame_rgb = frame

                frame = cv2.addWeighted(frame_rgb, 1.0, overlay, opacity, 0)

        self._show_image(frame)

    def _display_side_by_side(self):
        """Display side-by-side comparison"""
        # Get processed frame
        proc_frame = self.tiff_loader.get_frame(self.tiff_loader.current_frame, False)
        if proc_frame is None:
            return

        # Get unprocessed frame
        unproc_frame = self.tiff_loader.get_frame(self.tiff_loader.current_frame, True)
        if unproc_frame is None:
            unproc_frame = proc_frame

        # Create side-by-side image
        h, w = proc_frame.shape[:2]
        combined = np.zeros((h, w * 2, 3), dtype=np.uint8)

        # Processed on left
        if proc_frame.ndim == 2:
            combined[:, :w, :] = np.stack([proc_frame] * 3, axis=-1)
        else:
            combined[:, :w, :] = proc_frame

        # Unprocessed on right
        if unproc_frame.ndim == 2:
            combined[:, w:, :] = np.stack([unproc_frame] * 3, axis=-1)
        else:
            combined[:, w:, :] = unproc_frame

        # Add detection markers
        if self.tiff_loader.current_frame in self.detection_cache:
            _, overlay = self.detection_cache[self.tiff_loader.current_frame]
            if overlay is not None:
                opacity = self.opacity_slider.value() / 100.0
                combined[:, :w, :] = cv2.addWeighted(
                    combined[:, :w, :], 1.0, overlay, opacity, 0
                )

        self._show_image(combined)

    def _display_difference(self):
        """Display difference map"""
        proc_frame = self.tiff_loader.get_frame(self.tiff_loader.current_frame, False)
        unproc_frame = self.tiff_loader.get_frame(self.tiff_loader.current_frame, True)

        if proc_frame is None or unproc_frame is None:
            return

        # Convert to grayscale
        proc_gray = proc_frame if proc_frame.ndim == 2 else np.mean(proc_frame, axis=2)
        unproc_gray = (
            unproc_frame if unproc_frame.ndim == 2 else np.mean(unproc_frame, axis=2)
        )

        # Calculate absolute difference
        diff = np.abs(proc_gray.astype(float) - unproc_gray.astype(float))
        diff = (diff / diff.max() * 255).astype(np.uint8)

        # Colorize
        diff_color = np.zeros((diff.shape[0], diff.shape[1], 3), dtype=np.uint8)
        diff_color[:, :, 0] = diff  # Red channel

        self._show_image(diff_color)

    def _show_image(self, image: np.ndarray):
        """Display image in graphics view"""
        if image.dtype != np.uint8:
            image = (image / image.max() * 255).astype(np.uint8)

        height, width = image.shape[:2]
        bytes_per_line = 3 * width if image.ndim == 3 else width

        if image.ndim == 2:
            qimage = QImage(
                image.data,
                width,
                height,
                bytes_per_line,
                QImage.Format.Format_Grayscale8,
            )
        else:
            qimage = QImage(
                image.data, width, height, bytes_per_line, QImage.Format.Format_RGB888
            )

        pixmap = QPixmap.fromImage(qimage)
        self.graphics_scene.clear()
        self.graphics_scene.addPixmap(pixmap)
        self.graphics_scene.setSceneRect(QRectF(pixmap.rect()))
        self.graphics_view.fitInView(
            self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def _on_parameter_changed(self):
        """Handle parameter change with debounce"""
        self.param_timer.start(100)  # 100ms debounce
        self.tuning_indicator.setText("⚡ Live Tuning: Active...")

    def _on_parameter_changed_complete(self):
        """Update parameters and trigger detection"""
        self._update_params_from_ui()
        self._detect_current_frame()
        self.tuning_indicator.setText("⚡ Live Tuning: Inactive")

    def _update_params_from_ui(self):
        """Update params object from UI values"""
        for name, (slider, spinbox) in self.param_widgets.items():
            value = spinbox.value()
            if name == "threshold":
                self.params.threshold = int(value)
            elif name == "blur_radius":
                self.params.blur_radius = float(value)
            elif name == "min_area":
                self.params.min_area = int(value)
            elif name == "max_area":
                self.params.max_area = int(value)
            elif name == "min_aspect":
                self.params.min_aspect = float(value)
            elif name == "max_aspect":
                self.params.max_aspect = float(value)
            elif name == "min_solidity":
                self.params.min_solidity = float(value)

        self.params.marker_color = self.marker_color_combo.currentText()
        self.params.marker_shape = self.marker_shape_combo.currentText()
        self.params.marker_size = self.marker_size_slider.value()

    # Update _detect_current_frame to use worker.signals:
    def _detect_current_frame(self):
        """Detect in current frame only"""
        if self.tiff_loader.num_frames == 0 or self.is_processing:
            return

        frame_data = self.tiff_loader.get_frame(self.tiff_loader.current_frame, False)
        if frame_data is None:
            return

        self.status_label.setText("Detecting...")

        worker = DetectionWorker(
            frame_data, self.params, self.tiff_loader.current_frame
        )
        worker.signals.started.connect(
            lambda: self.detect_current_button.setEnabled(False)
        )
        worker.signals.finished.connect(self._on_detection_complete)
        worker.signals.error.connect(self._on_detection_error)

        self.thread_pool.start(worker)

    def _on_detection_complete(self, overlay: np.ndarray, result: dict):
        """Handle detection completion"""
        frame_idx = result["frame_idx"]
        detections = result["detections"]

        self.detection_cache[frame_idx] = (detections, overlay)
        self._update_display()
        self._update_frame_stats()
        self.detect_current_button.setEnabled(True)
        self.status_label.setText(f"Detection complete: {len(detections)} objects")

    def _on_detection_error(self, error_msg: str):
        """Handle detection error"""
        self.status_label.setText(f"Error: {error_msg}")
        self.detect_current_button.setEnabled(True)
        QMessageBox.critical(self, "Detection Error", error_msg)

    def _update_frame_stats(self):
        """Update frame statistics table"""
        if self.tiff_loader.current_frame not in self.detection_cache:
            self.frame_info_label.setText(f"Frame {self.tiff_loader.current_frame}")
            self.detection_count_label.setText("Detections: 0")
            self.avg_area_label.setText("Avg Area: -")
            self.avg_aspect_label.setText("Avg Aspect: -")
            self.avg_solidity_label.setText("Avg Solidity: -")
            self.log_table.setRowCount(0)
            return

        detections, _ = self.detection_cache[self.tiff_loader.current_frame]

        self.frame_info_label.setText(f"Frame {self.tiff_loader.current_frame}")
        self.detection_count_label.setText(f"Detections: {len(detections)}")

        if detections:
            avg_area = np.mean([d["area"] for d in detections])
            avg_aspect = np.mean([d["aspect"] for d in detections])
            avg_solidity = np.mean([d["solidity"] for d in detections])

            self.avg_area_label.setText(f"Avg Area: {avg_area:.1f} px²")
            self.avg_aspect_label.setText(f"Avg Aspect: {avg_aspect:.2f}")
            self.avg_solidity_label.setText(f"Avg Solidity: {avg_solidity:.3f}")
        else:
            self.avg_area_label.setText("Avg Area: -")
            self.avg_aspect_label.setText("Avg Aspect: -")
            self.avg_solidity_label.setText("Avg Solidity: -")

        # Update log table
        self.log_table.setRowCount(len(detections))
        for row, det in enumerate(detections):
            self.log_table.setItem(row, 0, QTableWidgetItem(str(det["id"])))
            self.log_table.setItem(row, 1, QTableWidgetItem(f"{det['area']:.0f}"))
            self.log_table.setItem(row, 2, QTableWidgetItem(f"{det['aspect']:.2f}"))
            self.log_table.setItem(row, 3, QTableWidgetItem(f"{det['solidity']:.3f}"))
            self.log_table.setItem(row, 4, QTableWidgetItem(str(det["x"])))
            self.log_table.setItem(row, 5, QTableWidgetItem(str(det["y"])))

        self.log_table.resizeColumnsToContents()

    def _update_summary(self):
        """Update stack summary statistics"""
        self.total_frames_label.setText(f"Total Frames: {self.tiff_loader.num_frames}")

        if self.detection_cache:
            frames_with_det = len(self.detection_cache)
            total_dets = sum(len(dets) for dets, _ in self.detection_cache.values())
            avg_per_frame = total_dets / frames_with_det

            self.frames_with_det_label.setText(
                f"Frames w/ Detections: {frames_with_det} ({100 * frames_with_det / self.tiff_loader.num_frames:.1f}%)"
            )
            self.total_dets_label.setText(f"Total Detections: {total_dets}")
            self.avg_per_frame_label.setText(f"Average/Frame: {avg_per_frame:.2f}")
        else:
            self.frames_with_det_label.setText("Frames w/ Detections: 0 (0%)")
            self.total_dets_label.setText("Total Detections: 0")
            self.avg_per_frame_label.setText("Average/Frame: 0")

    def _process_stack(self):
        """Process all frames in stack"""
        if self.tiff_loader.num_frames == 0 or self.is_processing:
            return

        self.is_processing = True
        self._update_ui_state()

        self.progress_bar.setVisible(True)
        self.progress_bar.setRange(0, self.tiff_loader.num_frames)

        self.status_label.setText("Processing stack...")

        for frame_idx in range(self.tiff_loader.num_frames):
            if not self.is_processing:
                break

            frame_data = self.tiff_loader.get_frame(frame_idx, False)
            if frame_data is not None:
                worker = DetectionWorker(frame_data, self.params, frame_idx)
                worker.signals.finished.connect(self._on_batch_detection_complete)
                worker.signals.error.connect(self._on_detection_error)
                self.thread_pool.start(worker)

            self.progress_bar.setValue(frame_idx + 1)
            QApplication.processEvents()

        self.is_processing = False
        self._update_ui_state()
        self.progress_bar.setVisible(False)
        self.status_label.setText("Stack processing complete")
        self._update_summary()

    def _on_batch_detection_complete(self, overlay: np.ndarray, result: dict):
        """Handle batch detection completion"""
        frame_idx = result["frame_idx"]
        detections = result["detections"]
        self.detection_cache[frame_idx] = (detections, overlay)

    def _cancel_processing(self):
        """Cancel ongoing processing"""
        self.is_processing = False
        self.thread_pool.clear()
        self.status_label.setText("Processing cancelled")
        self.progress_bar.setVisible(False)
        self._update_ui_state()

    def _toggle_playback(self):
        """Toggle movie playback"""
        if self.playback_timer.isActive():
            self.playback_timer.stop()
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPlay)
            )
        else:
            self.playback_timer.start(int(1000 / self.playback_fps))
            self.play_button.setIcon(
                self.style().standardIcon(QStyle.StandardPixmap.SP_MediaPause)
            )

    def _next_frame(self):
        """Go to next frame (for playback)"""
        next_frame = (self.tiff_loader.current_frame + 1) % self.tiff_loader.num_frames
        self._set_frame(next_frame)

    def _reset_zoom(self):
        """Reset zoom to fit"""
        self.graphics_view.fitInView(
            self.graphics_scene.sceneRect(), Qt.AspectRatioMode.KeepAspectRatio
        )

    def _update_ui_state(self):
        """Update UI element states based on current context"""
        has_file = self.tiff_loader.num_frames > 0
        is_idle = not self.is_processing

        self.detect_current_button.setEnabled(has_file and is_idle)
        self.process_stack_button.setEnabled(has_file and is_idle)
        self.cancel_button.setEnabled(not is_idle)

        # Parameter controls
        for slider, spinbox in self.param_widgets.values():
            slider.setEnabled(is_idle)
            spinbox.setEnabled(is_idle)

        self.marker_color_combo.setEnabled(is_idle)
        self.marker_shape_combo.setEnabled(is_idle)
        self.marker_size_slider.setEnabled(is_idle)
        self.opacity_slider.setEnabled(has_file)

        # Navigation
        self.first_frame_button.setEnabled(has_file)
        self.prev_frame_button.setEnabled(has_file)
        self.play_button.setEnabled(has_file)
        self.next_frame_button.setEnabled(has_file)
        self.last_frame_button.setEnabled(has_file)
        self.frame_spinbox.setEnabled(has_file)
        self.frame_slider.setEnabled(has_file)
        self.mode_combo.setEnabled(has_file)
        self.overlay_toggle.setEnabled(
            has_file and self.tiff_loader.overlay_data is not None
        )

    def _save_params(self):
        """Save current parameters to JSON"""
        import json

        path, _ = QFileDialog.getSaveFileName(
            self, "Save Parameters", "", "JSON Files (*.json)"
        )
        if path:
            with open(path, "w") as f:
                json.dump(self.params.__dict__, f, indent=2)
            self.status_label.setText(f"Parameters saved to {path}")

    def _save_snapshot(self):
        """Save current view as image"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Save Snapshot", "", "PNG Files (*.png)"
        )
        if path:
            pixmap = self.graphics_view.grab()
            pixmap.save(path)
            self.status_label.setText(f"Snapshot saved to {path}")

    def _export_csv(self):
        """Export detection results to CSV with enhanced parameters and contour data"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export CSV", "", "CSV Files (*.csv)"
        )
        if path:
            import csv

            with open(path, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame",
                        "detection_id",
                        "x",
                        "y",
                        "area",
                        "aspect",
                        "solidity",
                        "contour",
                        "threshold",
                        "blur_radius",
                        "min_area",
                        "max_area",
                        "min_aspect",
                        "max_aspect",
                        "min_solidity",
                    ]
                )
                for frame_idx, (detections, _) in self.detection_cache.items():
                    for det in detections:
                        writer.writerow(
                            [
                                frame_idx,
                                det["id"],
                                det["x"],
                                det["y"],
                                det["area"],
                                det["aspect"],
                                det["solidity"],
                                det.get("contour", ""),  # Contour data (serialized)
                                det["threshold"],
                                det["blur_radius"],
                                det["min_area"],
                                det["max_area"],
                                det["min_aspect"],
                                det["max_aspect"],
                                det["min_solidity"],
                            ]
                        )
            self.status_label.setText(f"CSV exported to {path}")

    def _export_video(self):
        """Export annotated video"""
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Video", "", "MP4 Files (*.mp4)"
        )
        if path:
            self.status_label.setText("Exporting video...")
            with imageio.get_writer(path, fps=5) as writer:
                for frame_idx in range(self.tiff_loader.num_frames):
                    if frame_idx in self.detection_cache:
                        _, overlay = self.detection_cache[frame_idx]
                        writer.append_data(overlay)
                    else:
                        frame = self.tiff_loader.get_frame(frame_idx, False)
                        if frame is not None:
                            writer.append_data(frame)
            self.status_label.setText(f"Video exported to {path}")

    def _apply_filter(self):
        """Apply frame filter"""
        filter_type = self.filter_combo.currentText()
        min_dets = self.min_det_spinbox.value()

        # This would filter the frame list in a real implementation
        self.status_label.setText(
            f"Filter applied: {filter_type}, Min detections: {min_dets}"
        )

    def _reset_filter(self):
        """Reset filters"""
        self.filter_combo.setCurrentIndex(0)
        self.min_det_spinbox.setValue(0)
        self.status_label.setText("Filters reset")

    def _show_about(self):
        """Show about dialog"""
        QMessageBox.about(
            self,
            "About Sperm Head Detection Workbench",
            "Mouse Sperm Head Detection Workbench\n\n"
            "A graphical tool for tuning detection parameters\n"
            "and visualizing results in real-time.\n\n",
        )


def main():
    """Main application entry point"""
    app = QApplication(sys.argv)

    # Set application style
    app.setStyle("Fusion")

    # Optional: Set dark palette
    # from PyQt6.QtGui import QPalette, QColor
    # palette = QPalette()
    # palette.setColor(QPalette.ColorRole.Window, QColor(53, 53, 53))
    # app.setPalette(palette)

    window = MainWindow()
    window.show()

    sys.exit(app.exec())


if __name__ == "__main__":
    main()
