import pandas as pd
from pathlib import Path
from src.pipeline import build_pipeline

def main():
    print("="*60)
    print("🏥 PIPELINE: CITAS CLÍNICAS (MESSY CLINIC APPOINTMENTS)")
    print("="*60)

    try:
        # 1. Cargar Datos
        raw_path = Path("data/raw/messy_clinic_appointments.csv")
        if not raw_path.exists():
            print(f"\n❌ ERROR: No se encontró el archivo en {raw_path}")
            return

        print("\n📥 Cargando datos crudos...")
        df_raw = pd.read_csv(raw_path)

        # 2. Mapeo de la Variable Objetivo
        print("\n🎯 Procesando variable objetivo (follow_up_required)...")
        # El dataset tiene inconsistencias: Yes, Y, 1 vs No, N, 0
        target_map = {'Y': 1, '1': 1, 'Yes': 1, 'N': 0, '0': 0, 'No': 0}
        df_raw['follow_up_bin'] = df_raw['follow_up_required'].map(target_map)
        
        # Separamos el target antes de que el pipeline deseche columnas sobrantes
        y = df_raw['follow_up_bin']
        X = df_raw.drop(columns=['follow_up_required', 'follow_up_bin'], errors='ignore')

        # 3. Aplicar Pipeline
        print("\n🏗️  Construyendo y aplicando el pipeline...")
        pipeline = build_pipeline()
        X_processed = pipeline.fit_transform(X)

        # Recuperar nombres de las columnas
        try:
            feature_names = pipeline.named_steps["preprocessor"].get_feature_names_out()
            clean_names = [name.split("__")[-1] for name in feature_names]
        except Exception:
            clean_names = [f"col_{i}" for i in range(X_processed.shape[1])]

        # 4. Ensamblar y Guardar
        print("\n💾 Guardando dataset procesado...")
        df_processed = pd.DataFrame(X_processed, columns=clean_names, index=X.index)
        
        # Volvemos a pegar la variable objetivo al final del dataset
        df_processed['follow_up_bin'] = y

        out_dir = Path("data/processed")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / "clean_clinic_appointments.csv"
        df_processed.to_csv(out_path, index=False)

        print("\n✅ PIPELINE COMPLETADO EXITOSAMENTE")
        print(f"📊 Dimensiones finales: {df_processed.shape[0]} filas × {df_processed.shape[1]} columnas")

    except Exception as e:
        print(f"\n❌ FATAL ERROR: El pipeline falló inesperadamente: {e}")

if __name__ == "__main__":
    main()