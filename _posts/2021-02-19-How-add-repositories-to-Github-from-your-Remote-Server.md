---
title: "Create a repository in Github from  a remote Server via SSH"
excerpt: " Create a repository in Github from  a remote Server via SSH"

header:
  image: "../assets/images/posts/2021-02-19-How-add-repositories-to-Github-from-your-Remote-Server/repo.jpg"
  teaser: "../assets/images/posts/2021-02-19-How-add-repositories-to-Github-from-your-Remote-Server/repo.jpg"
  caption: "Repo"
  

---

GitHub is a code hosting platform for version control and collaboration. It lets you and others work together on projects from anywhere. 

How to  create a **repository**  in Github  from  a **remote Server** with SSH?

A **repository** is usually used to organize a single project. Repositories can contain folders and files, images, videos, spreadsheets, and data sets – anything your project needs.



<img src="../assets/images/posts/2021-02-19-How-add-repositories-to-Github-from-your-Remote-Server/github.png" alt="github" style="zoom:50%;" />



## Your first time with git and github

If you’ve never used git or github before, and you want use them with your private server, there are a bunch of things that you need to do. It’s [very well explained on github](https://help.github.com/articles/set-up-git), but repeated here for completeness.

- Get a [github](https://github.com/) account.

- Download and install [git](https://git-scm.com/downloads). ( we assume that your server it has git installed)

- Set up git with your user name and email.

  - Open a terminal/shell in your remote server and type:

    ```
    git config --global user.name "Your name here"
    git config --global user.email "your_email@example.com"
    ```

    (Kep the `"  "`; during your setup of username and email.)

    I also do:

    ```
    git config --global color.ui true
    git config --global core.editor nano
    ```

    The first of these will enable colored output in the terminal; the second tells git that you want to use nano.

    

- Set up ssh on your computer. I like [Roger Peng](http://www.biostat.jhsph.edu/~rpeng)’s [guide to setting up password-less logins](http://www.biostat.jhsph.edu/bit/nopassword.html). Also see [github’s guide to generating SSH keys](https://help.github.com/articles/generating-ssh-keys).

  - Look to see if you have files `~/.ssh/id_rsa` and `~/.ssh/id_rsa.pub`.

  - If not, create such public/private keys: Open a terminal/shell and type:

    ```
    ssh-keygen -t rsa -C "your_email@example.com"
    ```

  - Copy your public key (the contents of the newly-created `id_rsa.pub` file) 

  - You can download the file .pub copy and delete

    **On a Windows**, in the terminal/shell, type:

    ```
    scp username@yourremoteserver.com:~/.ssh/id_rsa.pub C:\Users\username\Documents
    ```
  
    In case you have **Linux /Mac** you can use
    
    ```
    scp username@yourremoteserver.com:~/.ssh/id_rsa.pub  /Users/username/Documents/
    ```
    
    
  
- You open your any editor in your  Documents folder, copy the string and then go to the Github page.

- Paste your ssh public key into your github account settings.

  - Go to your github [Account Settings](https://github.com/settings/profile)

  - Click “[SSH Keys](https://github.com/settings/ssh)” on the left.

  - Click “Add SSH Key” on the right.

  - Add a label (like “My laptop”) and paste the public key into the big text box.

  - In a terminal/shell, type the following to test it:

    ```
    $ ssh -T git@github.com
    ```

  - If it says something like the following, it worked:

    ```
    Hi username! You've successfully authenticated, but Github does
    not provide shell access.
    ```

Don't forget delete your .pub file in your documents folder of your local computer.

```
del C:\Users\username\Documents\id_rsa.pub 
```

## Security Issues

If you got the error 

```
`@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
@         WARNING: UNPROTECTED PRIVATE KEY FILE!          @
@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
Permissions 0704 for ~/.ssh/id_rsa
```

That means that you have to change the permissions of your private key

Keys need to be only readable by you:

```
chmod 400 ~/.ssh/id_rsa
```

If Keys need to be read-writable by you:

```
chmod 600 ~/.ssh/id_rsa
```

*600* appears to be fine as well (in fact better in most cases, because you don't need to change file permissions later to edit it).

The relevant portion from the manpage (`man ssh`)

## Create new repo from local to Github

First we need to create a repository in the website of Github

1. In the upper-right corner of any page, use the drop-down menu, and select New **repository**.
2. Type a short, memorable name for your **repository**. ...
3. Optionally, add a description of your **repository**. ...
4. Choose a **repository** visibility. ...
5. Click **Create repository**.

Lets assume that your repository is called *newrepo* and you have a  project in your server.

Lets add this folder of your project to the Github 

1. Go to your terminal and enter to your remote server via ssh

   ```
   ssh username@yourremoteserver.com
   ```

    

2. Find the folder that you want to add to your  Github, lets says is named myproject,

   ```
   cd myproject
   ```

3. Write down the following commands:

```
echo "# newrepo" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote set-url origin git@github.com:username/newrepo.git
git push -u origin main
```

ad the end you got

```
Enter passphrase for key '/home/username/.ssh/id_rsa':
Counting objects: 52, done.
Delta compression using up to 2 threads.
Compressing objects: 100% (50/50), done.
Writing objects: 100% (52/52), 247.77 KiB, done.
Total 52 (delta 12), reused 0 (delta 0)
remote: Resolving deltas: 100% (12/12), done.
To git@github.com:username/newrepo.git
 * [new branch]      main -> main
Branch main set up to track remote branch main from origin.
```

if you go to your website of Github you will see you new published repo updated.

## Check setup

If you are interested to check if your setup is correct type

```
git config --list
```

and you got something like:

```
user.name=username
user.email=username@email.com
color.ui=true
core.editor=nano
core.repositoryformatversion=0
core.filemode=true
core.bare=false
core.logallrefupdates=true
remote.origin.url=git@github.com:username/newrepo.git
remote.origin.fetch=+refs/heads/*:refs/remotes/origin/*
branch.main.remote=origin
branch.main.merge=refs/heads/main
```

**Congratulation** we have  created  a repository in Github via  SSH  from your remote server.







