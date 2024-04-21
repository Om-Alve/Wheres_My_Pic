# Where's My Pic?

Where's My Pic? is a local image search engine powered by CLIP (Contrastive Language-Image Pre-training). It allows you to search for images in your local directories using natural language queries.

## Features

- Seamless integration with CLIP for powerful image-text retrieval.
- Easily index and update your local image directories.
- Perform natural language searches to find relevant images.
- Gradio-based user interface for a smooth searching experience.

## Some Sneak Peeks 

![Demo 1](assets/Screenshot%202024-04-07%20130309.png)
![Demo 2](assets/Screenshot%202024-04-07%20130512.png)
![Demo 3](assets/Screenshot%202024-04-07%20130600.png)

## Demo Video

Check out the demo video to see Where's My Pic? in action:

[![Demo Video](https://img.youtube.com/vi/oVJsJ0e6jWk/0.jpg)](https://www.youtube.com/watch?v=oVJsJ0e6jWk)

## Getting Started

1. Clone the repository:

   ```bash
   git clone https://github.com/Om-Alve/Wheres_My_Pic.git
   cd Wheres_My_Pic
   ```

2. Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

3. Download the CLIP model weights and place them in the `CLIP-VIT` directory.

4. Index your image folders:

   ```bash
   python build_db.py --add /path/to/image/folder1 /path/to/image/folder2
   ```

   This will create a Chroma database in the `img_db` directory for the first time it's executed and index the specified folders. You can use it to add new folders.

5. To add the changes to the indexed folders, for example adding or deleting a file use the following command

   ```bash
   python build_db.py --update
   ```
   This will update the existing database with the changes in the indexed folders.

6. Run the application:

   ```bash
   python app.py
   ```

   This will launch the Gradio interface, where you can enter natural language queries to search for images.

## Usage

1. Enter your search query in the text box (e.g., "a meme about machine learning").
2. The application will display the top results matching your query.
3. You can click on the images to view them in more detail.

## Contributing

Contributions are welcome! If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [Apache License](LICENSE).