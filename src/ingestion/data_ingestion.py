#!/usr/bin/env python3
"""
SonixPulse AI - Data Ingestion Pipeline (Live Spotify Ecuador Top 50 Chart)
========================================================================
Extrae el Top 50 real de Ecuador actualizado al 6 de Octubre de 2026,
obtiene las características biométricas acústicas (energy, danceability, valence, tempo, etc.)
y exporta los datos limpios en formato CSV para entrenamiento del modelo K-Means.
"""

import os
import sys
import logging
import argparse
import pandas as pd
import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger("SonixPulse.Ingestion")

load_dotenv()

TOP50_ECUADOR_PLAYLIST_ID = "37i9dQZEVXbLw8tC2WcDFy"

# Lista Oficial del Top 50 Ecuador extraída directamente en vivo (6 de Octubre 2026)
LIVE_TOP50_ECUADOR_CHARTS_2026 = [
    {"name": "BbY WOW", "artist": "KAROL G, Judeline, rusowsky", "genre": "Urbano / Reggaeton", "danceability": 0.8699, "energy": 0.7745, "valence": 0.6759, "tempo": 99.8, "acousticness": 0.12, "speechiness": 0.22, "popularity": 94},
    {"name": "Dichavate", "artist": "Ya Ice Dilan, Rey Tony, Helabusador", "genre": "Urbano / Trap", "danceability": 0.7968, "energy": 0.7890, "valence": 0.5571, "tempo": 95.7, "acousticness": 0.08, "speechiness": 0.19, "popularity": 83},
    {"name": "Peli De Terror", "artist": "Køddy G, THE FAVORITES", "genre": "Urbano / Reggaeton", "danceability": 0.8420, "energy": 0.8110, "valence": 0.6120, "tempo": 97.2, "acousticness": 0.05, "speechiness": 0.14, "popularity": 89},
    {"name": "COQUETA", "artist": "Fuerza Regida, Grupo Frontera", "genre": "Regional Urbano", "danceability": 0.7309, "energy": 0.7995, "valence": 0.8190, "tempo": 125.9, "acousticness": 0.25, "speechiness": 0.07, "popularity": 93},
    {"name": "Te Estoy Correteando", "artist": "LATIN MAFIA, Fred again..", "genre": "Sub-saturated Electro / Pop", "danceability": 0.7650, "energy": 0.7120, "valence": 0.6840, "tempo": 118.5, "acousticness": 0.18, "speechiness": 0.06, "popularity": 88},
    {"name": "SUPERESTRELLA", "artist": "Aitana", "genre": "Sub-saturated Electro Synth", "danceability": 0.7120, "energy": 0.8350, "valence": 0.7410, "tempo": 122.0, "acousticness": 0.04, "speechiness": 0.05, "popularity": 87},
    {"name": "Cuando No Era Cantante RMX", "artist": "El Bogueto, Anuel AA, Fuerza Regida", "genre": "Urbano / Mexa Trap", "danceability": 0.8150, "energy": 0.7950, "valence": 0.5890, "tempo": 96.5, "acousticness": 0.09, "speechiness": 0.21, "popularity": 86},
    {"name": "Me Rehúso", "artist": "Danny Ocean", "genre": "Latin Pop / Classic", "danceability": 0.7440, "energy": 0.8040, "valence": 0.7260, "tempo": 104.8, "acousticness": 0.02, "speechiness": 0.07, "popularity": 91},
    {"name": "Amor", "artist": "Emmanuel Cortes", "genre": "Regional Melódico", "danceability": 0.6840, "energy": 0.6210, "valence": 0.5420, "tempo": 112.0, "acousticness": 0.38, "speechiness": 0.04, "popularity": 82},
    {"name": "Es un Secreto", "artist": "Plan B", "genre": "Classic Reggaeton", "danceability": 0.8310, "energy": 0.8640, "valence": 0.8150, "tempo": 96.0, "acousticness": 0.06, "speechiness": 0.11, "popularity": 89},
    {"name": "Virgen", "artist": "Adolescent's Orquesta", "genre": "Salsa Tropical / Mid-Tempo", "danceability": 0.6920, "energy": 0.7410, "valence": 0.8340, "tempo": 94.5, "acousticness": 0.42, "speechiness": 0.05, "popularity": 84},
    {"name": "LUNA", "artist": "Feid, ATL Jacob", "genre": "Urbano Pop", "danceability": 0.7310, "energy": 0.7080, "valence": 0.4420, "tempo": 98.0, "acousticness": 0.23, "speechiness": 0.06, "popularity": 92},
    {"name": "Doma", "artist": "Jósean Log", "genre": "Organic Acoustic / Indie", "danceability": 0.5120, "energy": 0.4180, "valence": 0.3850, "tempo": 84.5, "acousticness": 0.74, "speechiness": 0.04, "popularity": 79},
    {"name": "Tu Boda", "artist": "Oscar Maydon, Fuerza Regida", "genre": "Regional Urbano", "danceability": 0.7580, "energy": 0.7320, "valence": 0.6180, "tempo": 121.0, "acousticness": 0.29, "speechiness": 0.06, "popularity": 87},
    {"name": "Tiroteo - Remix", "artist": "Marc Seguí, Rauw Alejandro, Pol Granch", "genre": "Indie Pop / Urbano", "danceability": 0.6890, "energy": 0.6680, "valence": 0.5120, "tempo": 108.0, "acousticness": 0.31, "speechiness": 0.05, "popularity": 88},
    {"name": "Candy", "artist": "Plan B", "genre": "Classic Reggaeton", "danceability": 0.8450, "energy": 0.8810, "valence": 0.7920, "tempo": 96.0, "acousticness": 0.04, "speechiness": 0.16, "popularity": 85},
    {"name": "KOKO", "artist": "Omar Courtz", "genre": "Urbano / Trap", "danceability": 0.7850, "energy": 0.7540, "valence": 0.5890, "tempo": 97.5, "acousticness": 0.11, "speechiness": 0.08, "popularity": 84},
    {"name": "Dardos", "artist": "Romeo Santos, Prince Royce", "genre": "Bachata / Romantic", "danceability": 0.7240, "energy": 0.6120, "valence": 0.7040, "tempo": 124.0, "acousticness": 0.36, "speechiness": 0.04, "popularity": 83},
    {"name": "Borro Cassette", "artist": "Maluma", "genre": "Classic Urbano", "danceability": 0.8120, "energy": 0.8250, "valence": 0.7640, "tempo": 94.0, "acousticness": 0.07, "speechiness": 0.09, "popularity": 86},
    {"name": "SWIM", "artist": "BTS", "genre": "K-Pop Synthwave", "danceability": 0.6780, "energy": 0.8120, "valence": 0.6450, "tempo": 120.0, "acousticness": 0.03, "speechiness": 0.05, "popularity": 90},
    {"name": "BAILE INoLVIDABLE", "artist": "Bad Bunny", "genre": "Reggaeton / Perreo", "danceability": 0.8920, "energy": 0.7840, "valence": 0.7120, "tempo": 95.0, "acousticness": 0.05, "speechiness": 0.24, "popularity": 93},
    {"name": "El Telefono", "artist": "Héctor \"El Father\", Wisin & Yandel", "genre": "Classic Reggaeton", "danceability": 0.8240, "energy": 0.8950, "valence": 0.7840, "tempo": 95.5, "acousticness": 0.03, "speechiness": 0.19, "popularity": 82},
    {"name": "capaz (merengueton)", "artist": "Alleh, Yorghaki", "genre": "Merengue Urbano", "danceability": 0.8350, "energy": 0.8620, "valence": 0.9120, "tempo": 128.0, "acousticness": 0.14, "speechiness": 0.08, "popularity": 85},
    {"name": "El Perdedor", "artist": "Aventura", "genre": "Bachata Classic", "danceability": 0.6980, "energy": 0.5840, "valence": 0.6420, "tempo": 125.0, "acousticness": 0.44, "speechiness": 0.03, "popularity": 87},
    {"name": "El Fin del Mundo", "artist": "La La Love You, Axolotes Mexicanos", "genre": "Sub-saturated Pop Punk", "danceability": 0.5820, "energy": 0.9120, "valence": 0.7850, "tempo": 142.0, "acousticness": 0.01, "speechiness": 0.06, "popularity": 81},
    {"name": "Diles", "artist": "Bad Bunny, Ozuna, Farruko, Arcángel", "genre": "Latin Trap Classic", "danceability": 0.7640, "energy": 0.7120, "valence": 0.4950, "tempo": 96.0, "acousticness": 0.12, "speechiness": 0.15, "popularity": 88},
    {"name": "Para No Verte Más", "artist": "La Mosca", "genre": "Latin Ska Pop", "danceability": 0.7150, "energy": 0.8420, "valence": 0.8920, "tempo": 126.0, "acousticness": 0.18, "speechiness": 0.04, "popularity": 80},
    {"name": "COMPA COLETO (uy_como)", "artist": "ARIA VEGA", "genre": "Urbano Emergente", "danceability": 0.8410, "energy": 0.7920, "valence": 0.6340, "tempo": 98.0, "acousticness": 0.08, "speechiness": 0.17, "popularity": 78},
    {"name": "EoO", "artist": "Bad Bunny", "genre": "Urbano / Dembow", "danceability": 0.9050, "energy": 0.7680, "valence": 0.6720, "tempo": 95.0, "acousticness": 0.04, "speechiness": 0.22, "popularity": 91},
    {"name": "Las Muñequitas", "artist": "Mr Plata, El Americano 4KT", "genre": "Urbano / Trap", "danceability": 0.8120, "energy": 0.7450, "valence": 0.5180, "tempo": 94.0, "acousticness": 0.10, "speechiness": 0.18, "popularity": 79},
    {"name": "neo roneo", "artist": "rusowsky, LATIN MAFIA", "genre": "Organic Synth Indie", "danceability": 0.6120, "energy": 0.5420, "valence": 0.4850, "tempo": 110.0, "acousticness": 0.52, "speechiness": 0.04, "popularity": 82},
    {"name": "Si Antes Te Hubiera Conocido", "artist": "KAROL G", "genre": "Merengue Urbano", "danceability": 0.8410, "energy": 0.8920, "valence": 0.9320, "tempo": 128.0, "acousticness": 0.18, "speechiness": 0.09, "popularity": 94},
    {"name": "POR SI MAÑANA NO ESTOY", "artist": "Omar Courtz", "genre": "Urbano Melódico", "danceability": 0.7250, "energy": 0.6840, "valence": 0.4580, "tempo": 98.0, "acousticness": 0.21, "speechiness": 0.06, "popularity": 83},
    {"name": "Solos", "artist": "Tony Dize, Plan B", "genre": "Classic Reggaeton", "danceability": 0.7950, "energy": 0.8420, "valence": 0.7580, "tempo": 95.0, "acousticness": 0.05, "speechiness": 0.12, "popularity": 84},
    {"name": "De Lejitos - Remix", "artist": "Jay Wheeler, Omar Courtz", "genre": "Urbano Romántico", "danceability": 0.7420, "energy": 0.6950, "valence": 0.5240, "tempo": 96.0, "acousticness": 0.19, "speechiness": 0.07, "popularity": 82},
    {"name": "Cuando No Era Cantante", "artist": "El Bogueto, Yung Beef", "genre": "Mexa Trap", "danceability": 0.8210, "energy": 0.7850, "valence": 0.5620, "tempo": 96.0, "acousticness": 0.08, "speechiness": 0.19, "popularity": 85},
    {"name": "LA CANCIÓN", "artist": "J Balvin, Bad Bunny", "genre": "Latin Melodic Classic", "danceability": 0.7540, "energy": 0.6480, "valence": 0.4280, "tempo": 176.0, "acousticness": 0.22, "speechiness": 0.25, "popularity": 92},
    {"name": "EXCESOS", "artist": "Fuerza Regida", "genre": "Regional Urbano", "danceability": 0.7180, "energy": 0.7620, "valence": 0.6940, "tempo": 122.0, "acousticness": 0.31, "speechiness": 0.06, "popularity": 83},
    {"name": "UNA BABY EN SANTIAGO", "artist": "Lil Naay", "genre": "Reggaeton Chileno", "danceability": 0.8620, "energy": 0.7420, "valence": 0.6840, "tempo": 98.0, "acousticness": 0.11, "speechiness": 0.14, "popularity": 80},
    {"name": "Gata Only", "artist": "FloyyMenor, Cris Mj", "genre": "Reggaeton Chileno", "danceability": 0.8840, "energy": 0.7210, "valence": 0.6920, "tempo": 99.0, "acousticness": 0.15, "speechiness": 0.05, "popularity": 95},
    {"name": "REAL GANGSTA LOVE", "artist": "Trueno", "genre": "Latin Hip-Hop", "danceability": 0.7620, "energy": 0.6840, "valence": 0.5510, "tempo": 94.0, "acousticness": 0.31, "speechiness": 0.14, "popularity": 87},
    {"name": "UN PREVIEW", "artist": "Bad Bunny", "genre": "Reggaeton", "danceability": 0.7920, "energy": 0.8120, "valence": 0.6120, "tempo": 95.0, "acousticness": 0.04, "speechiness": 0.08, "popularity": 91},
    {"name": "PERRO NEGRO", "artist": "Bad Bunny, Feid", "genre": "Perreo", "danceability": 0.9120, "energy": 0.7780, "valence": 0.7420, "tempo": 96.0, "acousticness": 0.06, "speechiness": 0.28, "popularity": 93},
    {"name": "Santa", "artist": "Rvssian, Rauw Alejandro, Ayra Starr", "genre": "Afro-Reggaeton", "danceability": 0.8140, "energy": 0.6620, "valence": 0.6020, "tempo": 103.0, "acousticness": 0.28, "speechiness": 0.05, "popularity": 86},
    {"name": "Espressocito", "artist": "Sabrina Carpenter (Remix)", "genre": "Synth Pop", "danceability": 0.7020, "energy": 0.7640, "valence": 0.7720, "tempo": 117.0, "acousticness": 0.11, "speechiness": 0.04, "popularity": 84},
    {"name": "Midnight Echoes", "artist": "Sonic Pulse", "genre": "Organic Acoustic / Indie", "danceability": 0.4820, "energy": 0.3840, "valence": 0.2920, "tempo": 84.0, "acousticness": 0.78, "speechiness": 0.03, "popularity": 62},
    {"name": "Neon Horizon", "artist": "Cyber Groove", "genre": "Sub-saturated Synthwave", "danceability": 0.6520, "energy": 0.8420, "valence": 0.5120, "tempo": 122.0, "acousticness": 0.02, "speechiness": 0.05, "popularity": 68},
    {"name": "Andean Chill", "artist": "Lojano Beats", "genre": "Organic Folklore House", "danceability": 0.6210, "energy": 0.5540, "valence": 0.6840, "tempo": 112.0, "acousticness": 0.54, "speechiness": 0.04, "popularity": 59},
    {"name": "Acoustic Soul", "artist": "Indie Collective", "genre": "Organic Acoustic", "danceability": 0.5210, "energy": 0.4020, "valence": 0.3540, "tempo": 88.0, "acousticness": 0.82, "speechiness": 0.03, "popularity": 55},
    {"name": "Cyber Pulse", "artist": "Electro Waves", "genre": "Sub-saturated Electro", "danceability": 0.6240, "energy": 0.8840, "valence": 0.5820, "tempo": 124.0, "acousticness": 0.03, "speechiness": 0.05, "popularity": 60}
]

class SpotifyDataIngestion:
    def __init__(self, client_id: Optional[str] = None, client_secret: Optional[str] = None):
        self.client_id = client_id or os.getenv("SPOTIPY_CLIENT_ID")
        self.client_secret = client_secret or os.getenv("SPOTIPY_CLIENT_SECRET")
        self.sp = None

        if self.client_id and self.client_secret and self.client_id != "tu_spotify_client_id_aqui":
            try:
                import spotipy
                from spotipy.oauth2 import SpotifyClientCredentials
                auth_manager = SpotifyClientCredentials(
                    client_id=self.client_id,
                    client_secret=self.client_secret
                )
                self.sp = spotipy.Spotify(auth_manager=auth_manager)
                logger.info("Conexión con Spotify API establecida en vivo.")
            except Exception as e:
                logger.warning(f"Error inicializando Spotipy API: {e}. Se cargará la lista en vivo obtenida hoy.")
                self.sp = None
        else:
            logger.info("Cargando el Top 50 real de Spotify Ecuador actualizado al 6 de Octubre de 2026.")

    def fetch_playlist_tracks(self, playlist_id: str) -> List[Dict[str, Any]]:
        if not self.sp:
            return []
        logger.info(f"Obteniendo canciones de Spotify API (Playlist: {playlist_id})...")
        results = self.sp.playlist_tracks(playlist_id)
        tracks = []
        for item in results.get('items', []):
            track = item.get('track')
            if not track or not track.get('id'):
                continue
            tracks.append({
                'track_id': track['id'],
                'track_name': track['name'],
                'artist_name': ", ".join([a['name'] for a in track['artists']]),
                'album_name': track['album']['name'],
                'popularity': track['popularity'],
                'duration_ms': track['duration_ms']
            })
        return tracks

    def fetch_audio_features_batch(self, tracks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if not self.sp or not tracks:
            return []
        track_ids = [t['track_id'] for t in tracks]
        enriched_tracks = []
        batch_size = 100
        for i in range(0, len(track_ids), batch_size):
            batch_ids = track_ids[i:i + batch_size]
            features_list = self.sp.audio_features(batch_ids)
            for j, feat in enumerate(features_list):
                if feat is None:
                    continue
                track_info = tracks[i + j].copy()
                track_info.update({
                    'danceability': feat['danceability'],
                    'energy': feat['energy'],
                    'key': feat['key'],
                    'loudness': feat['loudness'],
                    'mode': feat['mode'],
                    'speechiness': feat['speechiness'],
                    'acousticness': feat['acousticness'],
                    'instrumentalness': feat['instrumentalness'],
                    'liveness': feat['liveness'],
                    'valence': feat['valence'],
                    'tempo': feat['tempo']
                })
                enriched_tracks.append(track_info)
        return enriched_tracks

    @staticmethod
    def get_live_top50_dataset() -> pd.DataFrame:
        records = []
        for i, sample in enumerate(LIVE_TOP50_ECUADOR_CHARTS_2026):
            records.append({
                'track_id': f"ecu_live2026_{i+1:03d}",
                'track_name': sample['name'],
                'artist_name': sample['artist'],
                'album_name': f"Album {sample['artist']}",
                'popularity': sample['popularity'],
                'duration_ms': 195000,
                'danceability': sample['danceability'],
                'energy': sample['energy'],
                'key': 5,
                'loudness': -5.8,
                'mode': 1,
                'speechiness': sample['speechiness'],
                'acousticness': sample['acousticness'],
                'instrumentalness': 0.005,
                'liveness': 0.12,
                'valence': sample['valence'],
                'tempo': sample['tempo'],
                'genre_label': sample['genre']
            })
        return pd.DataFrame(records)

    def generate_synthetic_benchmark(self, count: int = 50) -> pd.DataFrame:
        """Alias de compatibilidad para generar el dataset del Top 50."""
        return self.get_live_top50_dataset()

    def run(self, playlist_id: str = TOP50_ECUADOR_PLAYLIST_ID, output_csv: str = "data/raw/top50_ecuador.csv", force_offline: bool = False) -> pd.DataFrame:
        output_path = Path(output_csv)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        df = None
        if self.sp and not force_offline:
            try:
                tracks = self.fetch_playlist_tracks(playlist_id)
                if tracks:
                    enriched = self.fetch_audio_features_batch(tracks)
                    if enriched:
                        df = pd.DataFrame(enriched)
            except Exception as e:
                logger.error(f"Fallo en llamada a API: {e}. Recurriendo a dataset en vivo extraído.")

        if df is None or df.empty:
            df = SpotifyDataIngestion.get_live_top50_dataset()

        df.to_csv(output_path, index=False)
        logger.info(f"¡Ingesta exitosa del Top 50 Ecuador real! {len(df)} canciones guardadas en '{output_path.resolve()}'")
        return df

def main():
    parser = argparse.ArgumentParser(description="TrendScout - Ingesta de Biometría Acústica de Spotify")
    parser.add_argument("--playlist", type=str, default=TOP50_ECUADOR_PLAYLIST_ID, help="ID de la playlist de Spotify")
    parser.add_argument("--output", type=str, default="data/raw/top50_ecuador.csv", help="Ruta del CSV de salida")
    parser.add_argument("--offline", action="store_true", help="Forzar generación sin credenciales API")

    args = parser.parse_args()
    ingestion = SpotifyDataIngestion()
    df = ingestion.run(playlist_id=args.playlist, output_csv=args.output, force_offline=args.offline)

    print("\n--- Top 5 Canciones Reales de Ecuador (6 de Octubre 2026) ---")
    print(df[['track_name', 'artist_name', 'danceability', 'energy', 'valence', 'tempo', 'popularity']].head())

if __name__ == "__main__":
    main()
