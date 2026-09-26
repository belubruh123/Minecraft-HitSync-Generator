"""Qt (PySide6) desktop app."""


def main(argv=None) -> int:
    import sys

    from PySide6.QtWidgets import QApplication

    from . import theme
    from .main_window import MainWindow

    app = QApplication.instance() or QApplication(sys.argv if argv is None else argv)
    app.setApplicationName("Hit-Sync")
    app.setOrganizationName("HitSync")
    theme.apply(app)
    win = MainWindow()
    win.show()
    args = [a for a in (sys.argv[1:] if argv is None else argv[1:]) if not a.startswith("-")]
    if args:
        win.drop_files(args)
    return app.exec()
