"""
Módulo para descargar videos de YouTube usando yt-dlp como librería Python.
"""

import os
from pathlib import Path

import yt_dlp


def download_video(url: str, output_dir: str, video_id: str = None) -> str:
    """
    Descarga un video de YouTube.

    Args:
        url: URL del video de YouTube
        output_dir: Directorio donde guardar el video
        video_id: ID opcional para nombrar el archivo

    Returns:
        Ruta al archivo descargado, o None si falla
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Usar el ID del video como nombre si no se proporciona uno
    if video_id is None:
        if "v=" in url:
            video_id = url.split("v=")[1].split("&")[0]
        elif "youtu.be/" in url:
            video_id = url.split("youtu.be/")[1].split("?")[0]
        else:
            video_id = "video"

    output_template = str(output_dir / f"{video_id}.%(ext)s")

    ydl_opts = {
        "format": "best[height<=1080][ext=mp4]/best[height<=1080]/best",
        "outtmpl": output_template,
        "noplaylist": True,
        "quiet": False,
        "no_warnings": False,
    }

    try:
        print(f"Descargando: {url}")

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])

        # Buscar el archivo descargado
        for ext in [".mp4", ".webm", ".mkv"]:
            video_path = output_dir / f"{video_id}{ext}"
            if video_path.exists():
                print(f"Video guardado en: {video_path}")
                return str(video_path)

        # Buscar cualquier archivo con el ID
        for file in output_dir.glob(f"{video_id}.*"):
            if file.suffix in [".mp4", ".webm", ".mkv"]:
                print(f"Video guardado en: {file}")
                return str(file)

        print(f"Advertencia: No se encontró el archivo descargado para {video_id}")
        return None

    except Exception as e:
        print(f"Error descargando {url}: {e}")
        return None


def download_videos_from_csv(
    csv_path: str, output_dir: str, url_column: str = "url"
) -> list:
    """
    Descarga múltiples videos desde un archivo CSV.

    Args:
        csv_path: Ruta al archivo CSV con URLs
        output_dir: Directorio de salida
        url_column: Nombre de la columna con las URLs

    Returns:
        Lista de rutas a los videos descargados
    """
    import pandas as pd

    df = pd.read_csv(csv_path)
    downloaded = []

    for idx, row in df.iterrows():
        url = row[url_column]
        video_id = f"video_{idx:04d}"

        video_path = download_video(url, output_dir, video_id)
        if video_path:
            downloaded.append(video_path)

    return downloaded


if __name__ == "__main__":
    # Ejemplo de uso
    test_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"
    download_video(test_url, "./test_downloads")
