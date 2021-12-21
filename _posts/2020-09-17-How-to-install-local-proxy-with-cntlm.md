---
title: "How to install a local proxy with cntlm"
excerpt: "How to Install Cntlm in Windows 10 and Linux"

header:
  image: "/assets/images/computer3.jpeg"
  teaser: "/assets/images/computer3.jpeg"
  caption: "We are all now connected by the Internet, like neurons in a giant brain. Stephen Hawking"
  
---


If you are working in an industry and your internet is an enterprise where all the traffic passes through your enterprise proxy and you need to use some applications where require internet and authentication requires a strong password. You can set up your local proxy with your strong password and use your local proxy to use it to use on several applications like a Virtual Machine or even We browsing like Firefox.


First we have to download Cntlm  for your operative system.

http://cntlm.sourceforge.net/

### Windows

After download.  Run setup.exe installer


## Edit cntlm.ini

After installation, you have to locate the configuration file. 

#### Linux

The default for Linux packages is **/etc/cntlm.conf**, for locally compiled source distribution ("./configure; make; make install") it's **/usr/local/etc/cntlm.conf** and 

#### Windows

for Windows installer it's **%PROGRAMFILES%\Cntlm\cntlm.ini** (usually X:\Program Files\Cntlm\cntlm.ini, where X is your system drive).

```
cd  "C:\Program Files (x86)"\Cntlm\
code cntlm.ini 
```

When you have found it, fire up your favorite editor (**not** a word processor) and open the file.



First a few rules, though - lines beginning with a hash, **#**, are *comments*: completely ignored. There is no required formatting and option names are case insensitive. Option values are parsed literally: a quote means a quote and is taken as part of the string, so do not quote, escape, etc. Anyway, you need to set these core options:

- **Username** - your domain/proxy account name
- **Domain** - the actual domain name
- **Workstation** - NetBIOS name of your workstation; Cntlm tries to autodetect it, but you might want to set it explicitly should dialect detection fail (see below)
- **Proxy** - IP address (or ping-able hostname) of your proxy; if you use several alternative proxies or know of backup ones, use this option multiple times; if one stops working, Cntlm will move on to the next
- **Listen** - local port number which Cntlm should bind to; the default is OK, but remember you can't have more than one application per port; you can use *netstat* to list used up ports (lines with LISTEN)



Next, we need to find out which NTLM dialect your proxy understands. It's a jungle out there and it can be quite challenging (i.e. boooring) to find a working NTLM setup - thank Bill.



```
Username ruslanmv
Domain YOURCOMPANYDOMAIN
Auth NTLM
PassLM EC6398A6D871d8B777E43632D37E2957
PassNT RF5EAAE6B9274EdCE8BC9B1589FD33F3
#PassNTLMv2 5E812882dC6FB537ACA3024C448E6B22    # Only for user 'ruslanmv', domain 'HOMEDOMAIN'
Proxy private_proxy.company.com:8080
NoProxy dev.company.com,*dev.company.com
Listen 3128
SOCKS5Proxy 3129
```





 Good thing Cntlm has this magic switch to do it for you - thank me. :) Save the configuration and run the following command; when asked, enter your proxy access password:

```
$ cntlm -I -M http://test.com
Config profile  1/11... OK (HTTP code: 200)
Config profile  2/11... OK (HTTP code: 200)
Config profile  3/11... OK (HTTP code: 200)
Config profile  4/11... OK (HTTP code: 200)
Config profile  5/11... OK (HTTP code: 200)
Config profile  6/11... Credentials rejected
Config profile  7/11... Credentials rejected
Config profile  8/11... OK (HTTP code: 200)
Config profile  9/11... OK (HTTP code: 200)
Config profile 10/11... OK (HTTP code: 200)
Config profile 11/11... OK (HTTP code: 200)
----------------------------[ Profile  0 ]------
Auth            NTLMv2
PassNTLMv2      4AC6525378DFc69CF6Bv234532943AC
------------------------------------------------
```

You see, **NTLMv2** - I told you to use it, now it's official. :) BTW, here you can see all tests running - it's just for demonstration purposes. Normal version finishes when it finds the first (i.e. most secure) working setup.



When you get your options (might be more than just **Auth** and **Pass\*** here), remove all previous password settings and paste the profile into the configuration file and save it. (Re)start Cntlm and it should work. To use it in your applications, replace the old proxy settings with "**localhost**", port same as you chose for **Listen**.

Visit http://cntlm.sf.net for HOWTO's and configuration tips.

## Start Cntlm



You can use Cntlm Start Menu shortcuts to start, stop and configure
the application. Cntlm is installed as an auto-start service.

OR:
Start -> Settings -> Control Panel -> Administrative Tools -> Services

OR (command line):
net start cntlm



```
cd  "C:\Program Files (x86)"\Cntlm\

```

```
C:\Program Files (x86)\Cntlm>net start cntlm
Servizio Cntlm Authentication Proxy in fase di avvio .
Avvio del servizio Cntlm Authentication Proxy riuscito.
```







## Check 

If you need to check from a command line try this:

```
sc query cntlm
```

This command should return something like this, if CNTLM is running:

```
SERVICE_NAME: cntlm
    TYPE               : 10  WIN32_OWN_PROCESS
    STATE              : 4  RUNNING
                            (STOPPABLE, NOT_PAUSABLE, IGNORES_SHUTDOWN)
    WIN32_EXIT_CODE    : 0  (0x0)
    SERVICE_EXIT_CODE  : 0  (0x0)
    CHECKPOINT         : 0x0
    WAIT_HINT          : 0x0
```

If stopped the result looks like this:

```
SERVICE_NAME: cntlm
    TYPE               : 10  WIN32_OWN_PROCESS
    STATE              : 1  STOPPED
    WIN32_EXIT_CODE    : 0  (0x0)
    SERVICE_EXIT_CODE  : 0  (0x0)
    CHECKPOINT         : 0x0
    WAIT_HINT          : 0x0
```

If you need to stop the service

```
net stop cntlm
```



## Additional Check



Install firefox, then go  to **Settings**>**General**> **Network Settings**

and add manual proxy configuration

for HTTP proxy

```
localhost 3128
```

check Also use this proxy for HTTPS

and  for SOCK HOST

```
localhost 3129
```
![](/assets/images/firefox.jpg)
then you can enter to google  and see how is working your custom proxy



## Uninstalling

Stop Cntlm service, run uninstaller from your Start Menu, or use
native Windows "Add/Remove Programs" Control Panel.

Congratulations! You have installed your local proxy.