"""
Script para subir modelo entrenado a Hugging Face Hub

Uso:
    conda activate french-solitaire
    python scripts/upload_to_hf.py --checkpoint checkpoints/dqn_final.pt --repo-id username/french-solitaire
"""
import argparse
import os
import sys
import shutil
from pathlib import Path

# Añadir directorio raíz al path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


def parse_args():
    """Parsea argumentos de línea de comandos."""
    parser = argparse.ArgumentParser(
        description="Sube modelo entrenado a Hugging Face Hub"
    )
    
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Ruta del checkpoint del modelo (.pt)",
    )
    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="ID del repositorio en HF Hub (formato: username/repo-name)",
    )
    parser.add_argument(
        "--token",
        type=str,
        default=None,
        help="Token de Hugging Face (o usar HF_TOKEN env var)",
    )
    parser.add_argument(
        "--private",
        action="store_true",
        help="Crear repositorio privado",
    )
    parser.add_argument(
        "--commit-message",
        type=str,
        default="Upload DQN model for French Solitaire",
        help="Mensaje del commit",
    )
    
    return parser.parse_args()


def prepare_upload_directory(checkpoint_path, temp_dir="./temp_hf_upload"):
    """
    Prepara un directorio temporal con todos los archivos a subir.
    
    Args:
        checkpoint_path (str): ruta del checkpoint
        temp_dir (str): directorio temporal
    
    Returns:
        str: ruta del directorio temporal
    """
    # Crear directorio temporal
    temp_path = Path(temp_dir)
    if temp_path.exists():
        shutil.rmtree(temp_path)
    temp_path.mkdir(parents=True)
    
    print(f"Preparando archivos en: {temp_path}")
    
    # Copiar checkpoint
    checkpoint_dest = temp_path / "pytorch_model.pt"
    shutil.copy(checkpoint_path, checkpoint_dest)
    print(f"  ✓ Checkpoint copiado: {checkpoint_dest.name}")
    
    # Copiar README de HF
    readme_src = Path("README_HF.md")
    readme_dest = temp_path / "README.md"
    if readme_src.exists():
        shutil.copy(readme_src, readme_dest)
        print(f"  ✓ README copiado: {readme_dest.name}")
    else:
        print(f"  ⚠ README_HF.md no encontrado, creando básico...")
        with open(readme_dest, "w") as f:
            f.write("# French Solitaire DQN Model\n\nDQN agent trained on French Solitaire (7x7).\n")
    
    # Copiar config
    config_src = Path("model_config.json")
    config_dest = temp_path / "config.json"
    if config_src.exists():
        shutil.copy(config_src, config_dest)
        print(f"  ✓ Config copiado: {config_dest.name}")
    
    # Copiar licencia
    license_src = Path("LICENSE")
    license_dest = temp_path / "LICENSE"
    if license_src.exists():
        shutil.copy(license_src, license_dest)
        print(f"  ✓ LICENSE copiado: {license_dest.name}")
    
    # Copiar código del agente y entorno (opcional, para reproducibilidad)
    code_dir = temp_path / "code"
    code_dir.mkdir()
    
    # Función para ignorar archivos/directorios innecesarios
    def ignore_patterns(directory, files):
        """Ignora __pycache__, .pyc, .pyo y otros archivos de cache."""
        return [
            f for f in files 
            if f == '__pycache__' 
            or f.endswith('.pyc') 
            or f.endswith('.pyo')
            or f.endswith('.pyd')
        ]
    
    # Copiar módulos principales
    for module in ["agent", "envs"]:
        module_src = Path(module)
        if module_src.exists():
            shutil.copytree(module_src, code_dir / module, ignore=ignore_patterns)
            print(f"  ✓ Código copiado: {module}/")
    
    # Copiar environment.yml
    env_src = Path("environment.yml")
    if env_src.exists():
        shutil.copy(env_src, temp_path / "environment.yml")
        print(f"  ✓ Environment.yml copiado")
    
    print(f"\nDirectorio preparado: {temp_path}")
    return str(temp_path)


def upload_to_hub(repo_id, folder_path, token=None, private=False, commit_message="Upload model"):
    """
    Sube el modelo a Hugging Face Hub.
    
    Args:
        repo_id (str): ID del repositorio (username/repo-name)
        folder_path (str): ruta del directorio con los archivos
        token (str): token de HF (opcional, auto-detecta si usaste `huggingface-cli login`)
        private (bool): si el repo es privado
        commit_message (str): mensaje del commit
    """
    try:
        from huggingface_hub import HfApi, create_repo
    except ImportError:
        print("\n❌ Error: huggingface_hub no instalado")
        print("Instalalo con: pip install huggingface-hub")
        sys.exit(1)
    
    # Obtener token (prioridad: argumento > env var > HF stored token)
    if token is None:
        token = os.getenv("HF_TOKEN")
    
    # Si no se proporciona token, HfApi automáticamente busca el token almacenado
    # por `huggingface-cli login` en ~/.cache/huggingface/token
    # No necesitamos hacer nada más, HfApi lo maneja automáticamente
    
    print("  ✓ Usando autenticación de Hugging Face")
    
    # Crear API (auto-detecta token almacenado si token=None)
    api = HfApi(token=token)
    
    print(f"\n📤 Subiendo modelo a: {repo_id}")
    
    # Crear repositorio (si no existe)
    try:
        create_repo(
            repo_id=repo_id,
            token=token,
            private=private,
            exist_ok=True,
            repo_type="model",
        )
        print(f"  ✓ Repositorio creado/verificado: {repo_id}")
    except Exception as e:
        print(f"  ⚠ Error al crear repositorio: {e}")
    
    # Limpiar archivos __pycache__ del repositorio (si existen)
    try:
        print("  🧹 Buscando archivos __pycache__ en el repositorio...")
        files = api.list_repo_files(repo_id, repo_type="model")
        pycache_files = [f for f in files if '__pycache__' in f or f.endswith('.pyc') or f.endswith('.pyo')]
        
        if pycache_files:
            print(f"  🗑️  Eliminando {len(pycache_files)} archivos de cache...")
            for file_path in pycache_files:
                api.delete_file(
                    path_in_repo=file_path,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"chore: remove {file_path}",
                    token=token,
                )
            print(f"  ✓ {len(pycache_files)} archivos de cache eliminados")
        else:
            print(f"  ✓ No hay archivos de cache para eliminar")
    except Exception as e:
        print(f"  ⚠ No se pudieron limpiar archivos de cache: {e}")
    
    # Subir archivos
    try:
        api.upload_folder(
            folder_path=folder_path,
            repo_id=repo_id,
            repo_type="model",
            commit_message=commit_message,
            token=token,
        )
        print(f"\n✅ Modelo subido exitosamente!")
        print(f"🔗 URL: https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"\n❌ Error al subir archivos: {e}")
        sys.exit(1)


def main():
    """Función principal."""
    args = parse_args()
    
    # Verificar que el checkpoint existe
    if not os.path.exists(args.checkpoint):
        print(f"❌ Error: checkpoint no encontrado: {args.checkpoint}")
        sys.exit(1)
    
    print("=== Subida a Hugging Face Hub ===")
    print(f"Checkpoint: {args.checkpoint}")
    print(f"Repo ID: {args.repo_id}")
    print(f"Privado: {args.private}")
    print()
    
    # Preparar directorio
    temp_dir = prepare_upload_directory(args.checkpoint)
    
    # Subir a HF Hub
    upload_to_hub(
        repo_id=args.repo_id,
        folder_path=temp_dir,
        token=args.token,
        private=args.private,
        commit_message=args.commit_message,
    )
    
    # Limpiar directorio temporal
    print(f"\n🧹 Limpiando directorio temporal: {temp_dir}")
    shutil.rmtree(temp_dir)
    
    print("\n✨ Proceso completado!")


if __name__ == "__main__":
    main()
