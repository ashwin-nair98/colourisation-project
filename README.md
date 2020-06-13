# Colorisation and Age Detection

Django server to colourise an image and detect age and gender of persons within picture.

## Setup
Steps to run locally:
1. Save the pre-trained model for age detection from [here](https://drive.google.com/file/d/1OuQx3ROwo0mgpGYsNXecHrheAaN2m2jN/view?ts=5ee10bd4) to `{project_root}/backend/pretrained_models/`
2. Save the pre-trained model for colourisation from [here](https://drive.google.com/file/d/11KtFescT_c2ANT_X-at0GHGx-oLxCgY0/view?usp=sharing) to `{project_root}/train/`
3. Ensure python version is 3.6. 
 `mkvirtualenv colorisation_server --python=python3`
 4. Install the requirements.
 `pip install -r requirements.txt`
 5. Run the server.
 `python manage.py runserver`
 6. Access at `http://localhost:8000/`

 Sample images stored at `sample_images` works best.