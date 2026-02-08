# Dev-Modus: Ordner von gui_tray.py = .../client
APP_ROOT = Path(__file__).resolve().parent
CONFIG_PATH = APP_ROOT / "config.json"
ICON_PATH = APP_ROOT / "icon.png"

"""
Läuft die komplette Kommunikation mit dem Server (inkl. Polling) im Hintergrund-Thread.
Bleibt die GUI währenddessen vollständig bedienbar.
Kannst du in der GUI mehrere Jobs nacheinander oder parallel zur Bearbeitung anderer Einstellungen anstoßen.
"""




