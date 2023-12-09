# unjumble-app
An app that answers question from the uploaded files


## Install Dependencies
```commandline
pip install -r requirements.txt
```

## Run the App
```commandline
python run.py
```

## Nginx
```commandline
sudo vi /etc/nginx/sites-available/unjumble-app
sudo nginx -t
sudo systemctl reload nginx
```

## Run the on the Server
```commandline
sudo ss -tulnp | grep :8080
sudo kill -9 xx
nohup python3 run.py &
sudo systemctl reload nginx
sudo lsof -ti:8080 | xargs kill -9
```

## Or simply use this to terminate all process at 8080 port
```commandline
sudo ss -tulnp | grep ':8080' | awk '{print $7}' | cut -d, -f1 | awk -F'"' '{print $2}' | xargs -r sudo kill -9
```
