import os

def collect_project_data(root_dir, output_file, ignore_dirs=None):
    if ignore_dirs is None:
        ignore_dirs = {'.venv', '.git'}  # Каталоги, которые нужно игнорировать

    with open(output_file, "w", encoding="utf-8") as out_file:
        def write_structure(current_dir, level=0):
            # Записать текущий каталог с отступом
            out_file.write(f"{'  ' * level}📁 {os.path.basename(current_dir)}/\n")

            try:
                # Получить списки файлов и папок в текущем каталоге
                entries = sorted(os.listdir(current_dir))  # Сортировка для предсказуемой структуры
                for entry in entries:
                    entry_path = os.path.join(current_dir, entry)

                    # Если это каталог, проверяем, нужно ли его игнорировать
                    if os.path.isdir(entry_path):
                        if entry in ignore_dirs:
                            continue
                        write_structure(entry_path, level + 1)
                    # Если это файл, записываем его
                    elif os.path.isfile(entry_path):
                        out_file.write(f"{'  ' * (level + 1)}📄 {entry}\n")

                        # Чтение текстового содержимого файлов с определенными расширениями
                        if entry.endswith(('.py', '.json', '.txt', '.md', '.env')):
                            out_file.write(f"{'  ' * (level + 2)}--- Содержимое файла ---\n")
                            try:
                                with open(entry_path, "r", encoding="utf-8") as file:
                                    content = file.read()
                                    out_file.write(f"{'  ' * (level + 2)}{content}\n")
                            except Exception as e:
                                out_file.write(f"{'  ' * (level + 2)}Ошибка чтения файла: {e}\n")
                            out_file.write(f"{'  ' * (level + 2)}--- Конец содержимого ---\n")
            except Exception as e:
                out_file.write(f"{'  ' * level}Ошибка доступа к каталогу: {e}\n")

        # Начать сбор данных с корневого каталога
        write_structure(root_dir)

if __name__ == "__main__":
    # Укажите путь к корневой папке проекта и имя выходного файла
    root_directory = "."  # Текущая директория
    output_filename = "project_summary.txt"

    collect_project_data(root_directory, output_filename)
    print(f"Сводка проекта сохранена в файл: {output_filename}")