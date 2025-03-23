ZIP_NAME="kashish"

mkdir -p "$ZIP_NAME"

# Preserve directory structure while copying files
rsync -War --delete --files-from='./zipass_files.txt' ./ "$ZIP_NAME/"
# Create a zip archive while preserving subfolder structure
zip -r "$ZIP_NAME.zip" "$ZIP_NAME"

# Clean up temporary directory
rm -rf "$ZIP_NAME"