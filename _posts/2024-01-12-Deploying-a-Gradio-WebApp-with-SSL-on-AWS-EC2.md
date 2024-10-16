---
title: "Deploying a Gradio WebApp with SSL on AWS EC2"
excerpt: "📜 Tutorial: Deploying a Gradio Microphone Recorder App with SSL on AWS EC2"

header:
  image: "./../assets/images/posts/2024-01-12-Deploying-a-Gradio-WebApp-with-SSL-on-AWS-EC2/caspar-camille-rubin-0qvBNep1Y04-unsplash.jpg"
  teaser: "./../assets/images/posts/2024-01-12-Deploying-a-Gradio-WebApp-with-SSL-on-AWS-EC2/caspar-camille-rubin-0qvBNep1Y04-unsplash.jpg"
  caption: "Cloud computing is really a no-brainer for any start-up because it allows you to test your business plan very quickly for little money"
  
---



Have you built an awesome Gradio web app with microphone recording, deployed it on an AWS EC2 instance, only to find out your browser won't let you use your mic?  You're not alone! This frustrating issue often pops up due to security restrictions that browsers place on microphone access.  The solution? **HTTPS**.

Modern web browsers require a secure connection (HTTPS) to allow access to sensitive hardware like microphones.  If your Gradio app is running on HTTP, your browser will likely block microphone access, leading to errors and a less than stellar user experience.

In this tutorial, we'll guide you through the process of securing your Gradio web app with HTTPS, enabling smooth and secure microphone recording. We'll cover everything from setting up an EC2 instance and deploying your Gradio app to configuring Nginx with a self-signed SSL certificate. By the end, you'll have a fully functional Gradio app with microphone recording capabilities, accessible securely over HTTPS.



## Table of Contents
1. [Prerequisites](#prerequisites)
2. [Step 1: Set Up an EC2 Instance](#step-1-set-up-an-ec2-instance)
3. [Step 2: Create the Gradio Microphone Recorder App](#step-2-create-the-gradio-microphone-recorder-app)
4. [Step 3: Install Nginx and Set Up SSL](#step-3-install-nginx-and-set-up-ssl)
5. [Step 4: Configure Security Groups](#step-4-configure-security-groups)
6. [Step 5: Access the App via HTTPS](#step-5-access-the-app-via-https)
7. [Conclusion](#conclusion)

---

## Prerequisites

Before you begin, ensure you have the following:

1. **An AWS EC2 instance** (Ubuntu 20.04 or 22.04 recommended).
2. **Basic knowledge of SSH** and command-line tools.
3. **A public IP address** of your EC2 instance.
4. **An SSH key pair** to access the EC2 instance.
5. **Security Group settings** configured to allow necessary traffic (more on this later).

---

## Step 1: Set Up an EC2 Instance

If you haven’t already set up an EC2 instance, follow these steps:

1. **Launch an EC2 instance** from the AWS Console:
   - Choose **Ubuntu 20.04/22.04** as the AMI.
   - Choose a **t2.micro** or **t3.micro** instance for free-tier use.
   
2. **Configure Security Group**:
   - Open the following ports:
     - `22` (SSH): For accessing the instance.
     - `80` (HTTP): For Nginx HTTP access (optional, we'll redirect to HTTPS).
     - `443` (HTTPS): For secure traffic.
     - `7860`: For Gradio’s local server (optional, Nginx will proxy traffic to this port).

3. **Launch the instance** and **SSH into it** using your private key:
   
   ```bash
   ssh -i /path/to/your-key.pem ubuntu@<your-ec2-ip-address>
   ```

---

## Step 2: Create the Gradio Microphone Recorder App

We will now create a simple Gradio app that records audio from your microphone.

1. **Install Python and Gradio**:

   Update your packages and install Python, Pip, and Gradio:

   ```bash
   sudo apt update
   sudo apt install python3-pip -y
   pip3 install gradio
   ```

2. **Create the Gradio app**:

   Create a file called `app.py`:

   ```bash
   nano app.py
   ```

   Paste the following code:

   ```python
   import gradio as gr

   def create_app():
       with gr.Blocks() as demo:
           gr.Markdown("# Microphone Test App\nRecord your message and play it back.")

           audio_input = gr.Audio(type="filepath", label="Record your message")
           audio_output = gr.Audio(label="Playback your recording")

           audio_input.change(fn=lambda x: x, inputs=audio_input, outputs=audio_output)

       return demo

   if __name__ == "__main__":
       app = create_app()
       app.launch(server_name="0.0.0.0", server_port=7860)
   ```

3. **Run the app**:

   Start the Gradio app:

   ```bash
   python3 app.py
   ```

   You should now be able to access the app via `http://<your-ec2-ip>:7860`.

---

## Step 3: Install Nginx and Set Up SSL

To secure the app, we will use **Nginx** as a reverse proxy and set up SSL with a self-signed certificate.

1. **Create the `setup.sh` Script**:

   This script installs Nginx, generates a self-signed SSL certificate, and configures Nginx to forward HTTPS traffic to your Gradio app.

   Create a file called `setup.sh`:

   ```bash
   nano setup.sh
   ```

   Paste the following code:

   ```bash
   #!/bin/bash

   # Step 1: Install Nginx
   sudo apt update -y
   sudo apt install nginx -y

   # Step 2: Generate a self-signed SSL certificate
   sudo mkdir -p /etc/nginx/ssl
   sudo openssl genrsa -out /etc/nginx/ssl/self-signed.key 2048
   sudo openssl req -new -x509 -key /etc/nginx/ssl/self-signed.key -out /etc/nginx/ssl/self-signed.crt -days 365 -subj "/CN=<your-ec2-ip>"

   # Step 3: Configure Nginx to use SSL and proxy to Gradio app
   sudo mv /etc/nginx/sites-available/default /etc/nginx/sites-available/default.bak
   sudo tee /etc/nginx/sites-available/default > /dev/null <<EOL
   server {
       listen 80;
       server_name <your-ec2-ip>;
       return 301 https://\$server_name\$request_uri;
   }

   server {
       listen 443 ssl;
       server_name <your-ec2-ip>;

       ssl_certificate /etc/nginx/ssl/self-signed.crt;
       ssl_certificate_key /etc/nginx/ssl/self-signed.key;

       location / {
           proxy_pass http://127.0.0.1:7860;
           proxy_set_header Host \$host;
           proxy_set_header X-Real-IP \$remote_addr;
           proxy_set_header X-Forwarded-For \$proxy_add_x_forwarded_for;
           proxy_set_header X-Forwarded-Proto \$scheme;
       }
   }
   EOL

   echo "Setup complete! Restart Nginx to apply changes: sudo systemctl restart nginx"
   ```

2. **Make the script executable and run it**:

   ```bash
   chmod +x setup.sh
   ./setup.sh
   ```

3. **Restart Nginx** to apply the changes:

   ```bash
   sudo systemctl restart nginx
   ```

   Nginx will now proxy requests to your Gradio app running on port `7860`, and you can access it via HTTPS.

---

## Step 4: Configure Security Groups

Ensure that the following ports are open in your EC2 instance’s **Security Group**:

- **Port 22 (SSH)**: For accessing your EC2 instance via SSH.
- **Port 80 (HTTP)**: For redirecting HTTP traffic to HTTPS (optional).
- **Port 443 (HTTPS)**: For secure HTTPS traffic.
- **Port 7860 (Gradio)**: For local access (optional, since Nginx is now handling this).

You can modify the Security Group settings in the **AWS Console** under the **Network & Security > Security Groups** section.

---

## Step 5: Access the App via HTTPS

Once Nginx is set up, and SSL is configured, you can access the app securely using the IP address of your EC2 instance:

```bash
https://<your-ec2-ip>
```

Since the SSL certificate is self-signed, your browser will likely show a security warning. You can bypass this warning by clicking on "Advanced" and proceeding to the site.

---



### Troubleshooting:

- If you still don’t see the Nginx welcome page, check if the port 80 or 443 is blocked in the EC2 security group.
- Make sure any firewall on the instance itself (e.g., `ufw`) is not blocking the connection.
- Verify the Nginx configuration file (`/etc/nginx/nginx.conf`) to ensure it's properly set up.



To test if Nginx is working properly on your EC2 instance, follow these steps:

### 1. **Check if Nginx is Installed and Running:**

   Run the following commands to check if Nginx is installed and the service is active:

   ```bash
   sudo systemctl status nginx
   ```

   This command will give you information about the status of Nginx. If it says "active (running)," then Nginx is running. If it's not running, start it using:

   ```bash
   sudo systemctl start nginx
   ```

   To ensure Nginx starts automatically on system reboot, run:

   ```bash
   sudo systemctl enable nginx
   ```

### 2. **Access Nginx Default Page:**

   Nginx installs a default page you can use to verify that it's running. To test this:

   1. **Get the public IP address of your EC2 instance** (if you haven't already).
   2. **Open a web browser** and go to `http://<your-ec2-public-ip>` (replace `<your-ec2-public-ip>` with your actual EC2 public IP).

   If Nginx is working properly, you should see a default page that says:

   ```
   Welcome to nginx!
   ```

   If you don't see this page, make sure the following is true:
   - **Nginx is running**.
   - **Port 80** is open in your EC2 instance's security group (this is the default HTTP port).
     - To check, go to your EC2 Dashboard, find **Security Groups** associated with your instance, and ensure **Inbound Rules** allow traffic on **port 80** (HTTP).

### 3. **Check Nginx Logs:**

   If you don't see the Nginx welcome page or encounter issues, you can check Nginx's error logs for troubleshooting.

   The default log files are located in `/var/log/nginx/`:

   - **Access logs**: `/var/log/nginx/access.log`
   - **Error logs**: `/var/log/nginx/error.log`

   You can view the logs using:

   ```bash
   sudo tail -f /var/log/nginx/error.log
   ```

   This will show you any errors that might be preventing Nginx from functioning correctly.

### 4. **Test Nginx Configuration:**

   If you've modified the Nginx configuration, it's a good idea to test the configuration file for syntax errors before reloading or restarting the service.

   Run the following command to test the configuration:

   ```bash
   sudo nginx -t
   ```

   If the output shows `syntax is ok` and `test is successful`, your Nginx configuration is correct.

### 5. **Restart Nginx After Configuration Changes:**

   If you've made changes to the Nginx configuration file (like setting up reverse proxy or SSL), restart Nginx to apply the changes:

   ```bash
   sudo systemctl restart nginx
   ```

### 6. **Test with HTTPS (If Configured):**

   If you've set up HTTPS with SSL (using Certbot or another tool), you can test it by going to `https://<your-ec2-public-ip>`. Make sure port **443** is also open in your EC2 security group for HTTPS traffic.



## Conclusion

In this tutorial, we walked through how to:

1. Set up an EC2 instance.
2. Build a simple Gradio microphone recorder app.
3. Install Nginx, configure a self-signed SSL certificate, and set up Nginx as a reverse proxy for the Gradio app.
4. Open the necessary ports and access the app via HTTPS.

This setup ensures your app is accessible securely, and you can expand on it by using a valid domain name and a Let's Encrypt SSL certificate in the future.

---

That's it! You should now have a fully functioning Gradio app running securely on AWS EC2. 🎉