import dpkt
import os
import numpy as np
from itertools import product
from scipy.stats import kurtosis, skew
import shutil

################################################################################
# Constants
DATASET_FOLDER = "./../data/output"
FEATURES_RESULT_PATH = "./../data/features"

################################################################################

def safe_stats(data):
    if len(data) > 0:
        return np.mean(data), np.std(data), np.var(data), np.amax(data), np.amin(data), kurtosis(data), skew(data)
    else:
        return np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan

def safe_percentile(data, percentile):
    if len(data) > 0:
        return np.percentile(data, percentile)
    else:
        return np.nan
    
################################################################################  
# Trancos top 1000 websites on 23/02/2025
website_mapping = {
  1: "google.com",
  2: "microsoft.com",
  3: "mail.ru",
  4: "facebook.com",
  5: "dzen.ru",
  6: "root-servers.net",
  7: "apple.com",
  8: "amazonaws.com",
  9: "youtube.com",
  10: "googleapis.com",
  11: "cloudflare.com",
  12: "akamai.net",
  13: "instagram.com",
  14: "twitter.com",
  15: "a-msedge.net",
  16: "gstatic.com",
  17: "office.com",
  18: "akamaiedge.net",
  19: "azure.com",
  20: "linkedin.com",
  21: "live.com",
  22: "tiktokcdn.com",
  23: "googletagmanager.com",
  24: "googlevideo.com",
  25: "amazon.com",
  26: "fbcdn.net",
  27: "akadns.net",
  28: "windowsupdate.com",
  29: "doubleclick.net",
  30: "wikipedia.org",
  31: "googleusercontent.com",
  32: "microsoftonline.com",
  33: "workers.dev",
  34: "l-msedge.net",
  35: "github.com",
  36: "t-msedge.net",
  37: "apple-dns.net",
  38: "bing.com",
  39: "trafficmanager.net",
  40: "wordpress.org",
  41: "office.net",
  42: "fastly.net",
  43: "whatsapp.net",
  44: "icloud.com",
  45: "googlesyndication.com",
  46: "netflix.com",
  47: "youtu.be",
  48: "aaplimg.com",
  49: "pinterest.com",
  50: "gtld-servers.net",
  51: "digicert.com",
  52: "sharepoint.com",
  53: "appsflyersdk.com",
  54: "yahoo.com",
  55: "domaincontrol.com",
  56: "whatsapp.com",
  57: "cloudfront.net",
  58: "goo.gl",
  59: "skype.com",
  60: "adobe.com",
  61: "vimeo.com",
  62: "windows.net",
  63: "spotify.com",
  64: "tiktokv.com",
  65: "cdn77.org",
  66: "msn.com",
  67: "s-msedge.net",
  68: "ntp.org",
  69: "gvt2.com",
  70: "ax-msedge.net",
  71: "bytefcdn-oversea.com",
  72: "google-analytics.com",
  73: "bit.ly",
  74: "roblox.com",
  75: "gvt1.com",
  76: "cloudflare.net",
  77: "wordpress.com",
  78: "tiktok.com",
  79: "nic.ru",
  80: "intuit.com",
  81: "e2ro.com",
  82: "wac-msedge.net",
  83: "zoom.us",
  84: "dual-s-msedge.net",
  85: "gandi.net",
  86: "x.com",
  87: "yandex.net",
  88: "mozilla.org",
  89: "qq.com",
  90: "ytimg.com",
  91: "tiktokrow-cdn.com",
  92: "edgekey.net",
  93: "unity3d.com",
  94: "opera.com",
  95: "blogspot.com",
  96: "office365.com",
  97: "cloudflare-dns.com",
  98: "reddit.com",
  99: "mts.ru",
  100: "samsung.com",
  101: "cdninstagram.com",
  102: "googleadservices.com",
  103: "baidu.com",
  104: "googledomains.com",
  105: "telekom.de",
  106: "msedge.net",
  107: "aiv-cdn.net",
  108: "a2z.com",
  109: "europa.eu",
  110: "snapchat.com",
  111: "bytefcdn-ttpeu.com",
  112: "wa.me",
  113: "health.mil",
  114: "amazon-adsystem.com",
  115: "nginx.org",
  116: "outlook.com",
  117: "t.me",
  118: "apache.org",
  119: "vk.com",
  120: "rocket-cdn.com",
  121: "rbxcdn.com",
  122: "sentry.io",
  123: "adnxs.com",
  124: "nginx.com",
  125: "dropbox.com",
  126: "github.io",
  127: "gravatar.com",
  128: "windows.com",
  129: "nist.gov",
  130: "app-analytics-services.com",
  131: "gccdn.net",
  132: "tiktokeu-cdn.com",
  133: "criteo.com",
  134: "spo-msedge.net",
  135: "dns.google",
  136: "app-measurement.com",
  137: "tumblr.com",
  138: "epicgames.com",
  139: "edgesuite.net",
  140: "userapi.com",
  141: "tiktokcdn-eu.com",
  142: "nih.gov",
  143: "okcdn.ru",
  144: "pki.goog",
  145: "macromedia.com",
  146: "xiaomi.com",
  147: "nytimes.com",
  148: "t-online.de",
  149: "msftncsi.com",
  150: "myfritz.net",
  151: "applovin.com",
  152: "paypal.com",
  153: "ggpht.com",
  154: "forms.gle",
  155: "archive.org",
  156: "telekom.net",
  157: "nflxso.net",
  158: "ui.com",
  159: "flickr.com",
  160: "one.one",
  161: "lencr.org",
  162: "ivi.ru",
  163: "dnsowl.com",
  164: "aliyuncs.com",
  165: "steamserver.net",
  166: "ttlivecdn.com",
  167: "medium.com",
  168: "meraki.com",
  169: "adriver.ru",
  170: "forbes.com",
  171: "b-msedge.net",
  172: "miit.gov.cn",
  173: "cnn.com",
  174: "mangosip.ru",
  175: "rubiconproject.com",
  176: "t.co",
  177: "soundcloud.com",
  178: "qlivecdn.com",
  179: "discord.gg",
  180: "registrar-servers.com",
  181: "theguardian.com",
  182: "w3.org",
  183: "trbcdn.net",
  184: "casalemedia.com",
  185: "akamaized.net",
  186: "tiktokcdn-us.com",
  187: "azurewebsites.net",
  188: "yandex.ru",
  189: "cdn-apple.com",
  190: "adobe.io",
  191: "android.com",
  192: "amazon.dev",
  193: "wildberries.ru",
  194: "wsdvs.com",
  195: "miui.com",
  196: "kaspersky.com",
  197: "shifen.com",
  198: "ebay.com",
  199: "demdex.net",
  200: "gmail.com",
  201: "bbc.co.uk",
  202: "bbc.com",
  203: "taboola.com",
  204: "amazonvideo.com",
  205: "doubleverify.com",
  206: "roku.com",
  207: "twitch.tv",
  208: "vtwenty.com",
  209: "creativecommons.org",
  210: "upcbroadband.com",
  211: "omtrdc.net",
  212: "b-cdn.net",
  213: "webex.com",
  214: "pv-cdn.net",
  215: "akamaihd.net",
  216: "ozon.ru",
  217: "sourceforge.net",
  218: "cisco.com",
  219: "mit.edu",
  220: "edgecdn.ru",
  221: "sciencedirect.com",
  222: "salesforce.com",
  223: "msftconnecttest.com",
  224: "comcast.net",
  225: "3gppnetwork.org",
  226: "imdb.com",
  227: "scorecardresearch.com",
  228: "cedexis.net",
  229: "jomodns.com",
  230: "researchgate.net",
  231: "ubuntu.com",
  232: "example.com",
  233: "shopify.com",
  234: "nvidia.com",
  235: "who.int",
  236: "my.com",
  237: "youtube-nocookie.com",
  238: "arubanetworks.com",
  239: "ksyuncdn.com",
  240: "azureedge.net",
  241: "hubspot.com",
  242: "openai.com",
  243: "linktr.ee",
  244: "canva.com",
  245: "doi.org",
  246: "discord.com",
  247: "nr-data.net",
  248: "vungle.com",
  249: "googleblog.com",
  250: "avast.com",
  251: "opendns.com",
  252: "hp.com",
  253: "facebook.net",
  254: "pubmatic.com",
  255: "oracle.com",
  256: "wixsite.com",
  257: "slack.com",
  258: "cdngslb.com",
  259: "byteoversea.net",
  260: "azurefd.net",
  261: "tinyurl.com",
  262: "launchdarkly.com",
  263: "issuu.com",
  264: "crashlytics.com",
  265: "openx.net",
  266: "ampproject.org",
  267: "wikimedia.org",
  268: "ttdns2.com",
  269: "appsflyer.com",
  270: "2mdn.net",
  271: "mtgglobals.com",
  272: "taobao.com",
  273: "booking.com",
  274: "cdc.gov",
  275: "weebly.com",
  276: "reuters.com",
  277: "netangels.ru",
  278: "f5.com",
  279: "drom.ru",
  280: "3lift.com",
  281: "google.com.br",
  282: "reg.ru",
  283: "washingtonpost.com",
  284: "harvard.edu",
  285: "adsrvr.org",
  286: "cpanel.net",
  287: "stripe.com",
  288: "tiktokv.us",
  289: "samsungcloud.com",
  290: "ripn.net",
  291: "vedcdnlb.com",
  292: "php.net",
  293: "www.gov.uk",
  294: "chatgpt.com",
  295: "steampowered.com",
  296: "pangle.io",
  297: "espn.com",
  298: "alibabadns.com",
  299: "tradingview.com",
  300: "dailymail.co.uk",
  301: "mzstatic.com",
  302: "samsungqbe.com",
  303: "amazon.co.uk",
  304: "godaddy.com",
  305: "appcenter.ms",
  306: "bidswitch.net",
  307: "slideshare.net",
  308: "mi.com",
  309: "ibm.com",
  310: "sharethrough.com",
  311: "wiley.com",
  312: "inmobi.com",
  313: "playstation.net",
  314: "jsdelivr.net",
  315: "xcal.tv",
  316: "ea.com",
  317: "wsj.com",
  318: "nflxvideo.net",
  319: "weather.com",
  320: "outbrain.com",
  321: "branch.io",
  322: "salesforceliveagent.com",
  323: "go.com",
  324: "smartadserver.com",
  325: "1c.ru",
  326: "dailymotion.com",
  327: "weibo.com",
  328: "ttvnw.net",
  329: "nature.com",
  330: "twimg.com",
  331: "launchpad.net",
  332: "atomile.com",
  333: "adtrafficquality.google",
  334: "bloomberg.com",
  335: "etsy.com",
  336: "shalltry.com",
  337: "googletagservices.com",
  338: "allawnos.com",
  339: "forter.com",
  340: "debian.org",
  341: "samsungcloudsolution.com",
  342: "springer.com",
  343: "adsafeprotected.com",
  344: "fastly-edge.com",
  345: "un.org",
  346: "dnsmadeeasy.com",
  347: "moe.video",
  348: "gnu.org",
  349: "stackoverflow.com",
  350: "vkuser.net",
  351: "liadm.com",
  352: "wp.com",
  353: "static.microsoft",
  354: "mcafee.com",
  355: "nease.net",
  356: "checkpoint.com",
  357: "google.de",
  358: "douyincdn.com",
  359: "gcdn.co",
  360: "id5-sync.com",
  361: "clarity.ms",
  362: "sc-cdn.net",
  363: "businessinsider.com",
  364: "zendesk.com",
  365: "notamedia.ru",
  366: "goskope.com",
  367: "gosuslugi.ru",
  368: "ring.com",
  369: "spov-msedge.net",
  370: "pixabay.com",
  371: "worldfcdn2.com",
  372: "rutube.ru",
  373: "cdn20.com",
  374: "inner-active.mobi",
  375: "stanford.edu",
  376: "yximgs.com",
  377: "independent.co.uk",
  378: "list-manage.com",
  379: "duckduckgo.com",
  380: "huawei.com",
  381: "pages.dev",
  382: "braze.com",
  383: "nasa.gov",
  384: "heytapdl.com",
  385: "media-amazon.com",
  386: "t-mobile.com",
  387: "cnbc.com",
  388: "unsplash.com",
  389: "mynetname.net",
  390: "capcutapi.com",
  391: "kaspersky-labs.com",
  392: "deepintent.com",
  393: "tp-link.com",
  394: "xboxlive.com",
  395: "aliexpress.com",
  396: "ibyteimg.com",
  397: "foxnews.com",
  398: "sohu.com",
  399: "duckdns.org",
  400: "autodesk.com",
  401: "eset.com",
  402: "elasticbeanstalk.com",
  403: "quora.com",
  404: "xhamster.com",
  405: "consultant.ru",
  406: "globo.com",
  407: "alibaba.com",
  408: "indexww.com",
  409: "ok.ru",
  410: "cloud.microsoft",
  411: "amazonalexa.com",
  412: "ngenix.net",
  413: "indeed.com",
  414: "google.co.uk",
  415: "statista.com",
  416: "npr.org",
  417: "eventbrite.com",
  418: "telegraph.co.uk",
  419: "bilibili.com",
  420: "netflix.net",
  421: "force.com",
  422: "temu.com",
  423: "online-metrix.net",
  424: "rambler.ru",
  425: "byteoversea.com",
  426: "dell.com",
  427: "avcdn.net",
  428: "nstld.com",
  429: "goodreads.com",
  430: "telegram.org",
  431: "myshopify.com",
  432: "amazon.de",
  433: "netease.com",
  434: "hichina.com",
  435: "giphy.com",
  436: "wbx2.com",
  437: "walmart.com",
  438: "kwai-pro.com",
  439: "sfx.ms",
  440: "gamepass.com",
  441: "mozilla.com",
  442: "yahoo.co.jp",
  443: "sophos.com",
  444: "firetvcaptiveportal.com",
  445: "teamviewer.com",
  446: "xserver.jp",
  447: "imgsmail.ru",
  448: "usatoday.com",
  449: "naver.com",
  450: "media.net",
  451: "nflximg.com",
  452: "time.com",
  453: "dnspod.net",
  454: "linkos.bg",
  455: "g.page",
  456: "aliyun.com",
  457: "telegram.me",
  458: "safebrowsing.apple",
  459: "addtoany.com",
  460: "grammarly.com",
  461: "mikrotik.com",
  462: "wired.com",
  463: "ys7.com",
  464: "speedtest.net",
  465: "duolingo.com",
  466: "mega.co.nz",
  467: "kwai.net",
  468: "ctdns.cn",
  469: "behance.net",
  470: "indiatimes.com",
  471: "sc-gw.com",
  472: "avsxappcaptiveportal.com",
  473: "amzn.to",
  474: "entrust.net",
  475: "mediatek.com",
  476: "scribd.com",
  477: "beian.gov.cn",
  478: "dns-parking.com",
  479: "hicloud.com",
  480: "aol.com",
  481: "myhuaweicloud.com",
  482: "blogger.com",
  483: "easebar.com",
  484: "calendly.com",
  485: "name-services.com",
  486: "grammarly.io",
  487: "cmediahub.ru",
  488: "cnet.com",
  489: "uber.com",
  490: "creativecdn.com",
  491: "alicdn.com",
  492: "comfortel.pro",
  493: "mysql.com",
  494: "uol.com.br",
  495: "timeweb.ru",
  496: "360yield.com",
  497: "wix.com",
  498: "squarespace.com",
  499: "imcmdb.net",
  500: "ovscdns.com",
  501: "ft.com",
  502: "cdnbuild.net",
  503: "verisign.com",
  504: "trendmicro.com",
  505: "surveymonkey.com",
  506: "shein.com",
  507: "icloud-content.com",
  508: "rlcdn.com",
  509: "disqus.com",
  510: "ietf.org",
  511: "nypost.com",
  512: "awswaf.com",
  513: "intel.com",
  514: "gitlab.com",
  515: "tds.net",
  516: "comcast.com",
  517: "tencent-cloud.net",
  518: "sentinelone.net",
  519: "rzone.de",
  520: "criteo.net",
  521: "ted.com",
  522: "rakuten.co.jp",
  523: "google.ca",
  524: "deviantart.com",
  525: "vkontakte.ru",
  526: "paloaltonetworks.com",
  527: "hotjar.com",
  528: "steamcommunity.com",
  529: "imgur.com",
  530: "xerox.com",
  531: "binance.com",
  532: "onetrust.com",
  533: "google.fr",
  534: "msidentity.com",
  535: "yahoodns.net",
  536: "visualstudio.com",
  537: "marriott.com",
  538: "adobe.net",
  539: "vivoglobal.com",
  540: "virtualearth.net",
  541: "fast.com",
  542: "patreon.com",
  543: "ca.gov",
  544: "plesk.com",
  545: "buzzfeed.com",
  546: "line.me",
  547: "ups.com",
  548: "playstation.com",
  549: "ipv4only.arpa",
  550: "jquery.com",
  551: "spaceweb.pro",
  552: "loc.gov",
  553: "okta.com",
  554: "mailchimp.com",
  555: "amazontrust.com",
  556: "oup.com",
  557: "aboutads.info",
  558: "target.com",
  559: "tripadvisor.com",
  560: "googlezip.net",
  561: "yandex.com",
  562: "conviva.com",
  563: "trustpilot.com",
  564: "att.net",
  565: "berkeley.edu",
  566: "tandfonline.com",
  567: "britannica.com",
  568: "atlassian.com",
  569: "cornell.edu",
  570: "akam.net",
  571: "ezvizlife.com",
  572: "smaato.net",
  573: "cbsnews.com",
  574: "hostgator.com",
  575: "merriam-webster.com",
  576: "amplitude.com",
  577: "shutterstock.com",
  578: "cookiedatabase.org",
  579: "crpt.ru",
  580: "oraclecloud.com",
  581: "onlinepbx.ru",
  582: "myqcloud.com",
  583: "sapo.pt",
  584: "usgovcloudapi.net",
  585: "amazon.co.jp",
  586: "rt.ru",
  587: "docker.com",
  588: "qualtrics.com",
  589: "licdn.com",
  590: "accuweather.com",
  591: "chinamobile.com",
  592: "froggydelight.com",
  593: "freepik.com",
  594: "lijit.com",
  595: "livejournal.com",
  596: "google.co.jp",
  597: "wattpad.com",
  598: "mozgcp.net",
  599: "techcrunch.com",
  600: "yelp.com",
  601: "ikea.com",
  602: "coinmarketcap.com",
  603: "sina.com.cn",
  604: "rackspace.net",
  605: "newrelic.com",
  606: "dzeninfra.ru",
  607: "google.es",
  608: "cookielaw.org",
  609: "impervadns.net",
  610: "ovscdns.net",
  611: "nel.goog",
  612: "bdydns.com",
  613: "datadoghq.com",
  614: "yieldmo.com",
  615: "nike.com",
  616: "quantserve.com",
  617: "163.com",
  618: "agkn.com",
  619: "adjust.com",
  620: "fontawesome.com",
  621: "samsungapps.com",
  622: "stackadapt.com",
  623: "live.net",
  624: "ieee.org",
  625: "optimizely.com",
  626: "fandom.com",
  627: "tapad.com",
  628: "fb.com",
  629: "wb.ru",
  630: "svc.ms",
  631: "discord.media",
  632: "latimes.com",
  633: "xiaomi.net",
  634: "theverge.com",
  635: "erome.com",
  636: "byteglb.com",
  637: "dotomi.com",
  638: "w3schools.com",
  639: "elpais.com",
  640: "noaa.gov",
  641: "crwdcntrl.net",
  642: "newsweek.com",
  643: "samsungacr.com",
  644: "typekit.net",
  645: "mayoclinic.org",
  646: "free.fr",
  647: "warnerbros.com",
  648: "google.it",
  649: "delfi.lt",
  650: "google.co.in",
  651: "pinimg.com",
  652: "bitrix24.ru",
  653: "360safe.com",
  654: "adobedtm.com",
  655: "webmd.com",
  656: "klaviyo.com",
  657: "threads.net",
  658: "worldnic.com",
  659: "aws.dev",
  660: "myspace.com",
  661: "service.gov.uk",
  662: "cdnhwc1.com",
  663: "cdnvideo.ru",
  664: "heytapmobile.com",
  665: "jotform.com",
  666: "typeform.com",
  667: "prnewswire.com",
  668: "investopedia.com",
  669: "herokudns.com",
  670: "amazon.ca",
  671: "browser-intake-datadoghq.com",
  672: "nbcnews.com",
  673: "appspot.com",
  674: "brave.com",
  675: "cambridge.org",
  676: "bidr.io",
  677: "dynatrace.com",
  678: "amazon.fr",
  679: "dbankcloud.com",
  680: "cqloud.com",
  681: "mtu.ru",
  682: "hbr.org",
  683: "bestbuy.com",
  684: "33across.com",
  685: "avito.ru",
  686: "imrworldwide.com",
  687: "coingecko.com",
  688: "xvideos.com",
  689: "pvp.net",
  690: "vidaahub.com",
  691: "intercom.io",
  692: "redislabs.com",
  693: "1rx.io",
  694: "redhat.com",
  695: "footprintdns.com",
  696: "bamgrid.com",
  697: "fedex.com",
  698: "maricopa.gov",
  699: "zemanta.com",
  700: "adform.net",
  701: "homedepot.com",
  702: "awsglobalaccelerator.com",
  703: "dns.jp",
  704: "azure-dns.com",
  705: "herokuapp.com",
  706: "mediafire.com",
  707: "google.com.hk",
  708: "riotgames.com",
  709: "sberdevices.ru",
  710: "usda.gov",
  711: "washington.edu",
  712: "sberbank.ru",
  713: "gumgum.com",
  714: "anydesk.com",
  715: "dns-shop.ru",
  716: "healthline.com",
  717: "telephony.goog",
  718: "bugsnag.com",
  719: "e-msedge.net",
  720: "everesttech.net",
  721: "withgoogle.com",
  722: "cloudapp.net",
  723: "vivo.com.cn",
  724: "shopee.co.id",
  725: "yellowblue.io",
  726: "sagepub.com",
  727: "unesco.org",
  728: "lemonde.fr",
  729: "contextweb.com",
  730: "airbnb.com",
  731: "allaboutcookies.org",
  732: "people.com",
  733: "linode.com",
  734: "cdnhwc2.com",
  735: "digitalocean.com",
  736: "rncdn7.com",
  737: "miwifi.com",
  738: "clever.com",
  739: "ya.ru",
  740: "service-now.com",
  741: "fda.gov",
  742: "shopee.com.br",
  743: "tbcache.com",
  744: "line-apps.com",
  745: "networkadvertising.org",
  746: "mixpanel.com",
  747: "amazon.in",
  748: "rackspace.com",
  749: "box.com",
  750: "ks-cdn.com",
  751: "mobile.de",
  752: "name.com",
  753: "rbc.ru",
  754: "a-mo.net",
  755: "zillow.com",
  756: "cloudflareinsights.com",
  757: "usps.com",
  758: "turn.com",
  759: "discordapp.com",
  760: "express.co.uk",
  761: "exp-tas.com",
  762: "incapdns.net",
  763: "omnitagjs.com",
  764: "fiverr.com",
  765: "nba.com",
  766: "cox.net",
  767: "att.com",
  768: "pccc.com",
  769: "immedia-semi.com",
  770: "bandcamp.com",
  771: "mgid.com",
  772: "msecnd.net",
  773: "wyzecam.com",
  774: "onelink.me",
  775: "google.cn",
  776: "githubusercontent.com",
  777: "ioam.de",
  778: "substack.com",
  779: "nationalgeographic.com",
  780: "theatlantic.com",
  781: "tremorhub.com",
  782: "2gis.com",
  783: "heylink.me",
  784: "ssl-images-amazon.com",
  785: "quickconnect.to",
  786: "onlyfans.com",
  787: "xbox.com",
  788: "steamstatic.com",
  789: "dribbble.com",
  790: "hihonorcloud.com",
  791: "pendo.io",
  792: "kargo.com",
  793: "themeforest.net",
  794: "huffingtonpost.com",
  795: "news.com.au",
  796: "change.org",
  797: "mgts.ru",
  798: "amazon.it",
  799: "life360.com",
  800: "me.com",
  801: "recaptcha.net",
  802: "istockphoto.com",
  803: "viber.com",
  804: "apigee.net",
  805: "princeton.edu",
  806: "pbs.org",
  807: "nest.com",
  808: "360.cn",
  809: "state.gov",
  810: "dbankcloud.cn",
  811: "unrulymedia.com",
  812: "wswebcdn.com",
  813: "globalsign.com",
  814: "stickyadstv.com",
  815: "teads.tv",
  816: "attn.tv",
  817: "protek.ru",
  818: "liftoff.io",
  819: "xiaohongshu.com",
  820: "go-mpulse.net",
  821: "google.pl",
  822: "scdn.co",
  823: "cloudns.net",
  824: "iso.org",
  825: "isappcloud.com",
  826: "onetag-sys.com",
  827: "dns.ne.jp",
  828: "mailchi.mp",
  829: "perfectdomain.com",
  830: "allegro.pl",
  831: "whitehouse.gov",
  832: "lenovo.com",
  833: "adobedc.net",
  834: "ameblo.jp",
  835: "worldbank.org",
  836: "lgtvcommon.com",
  837: "dnsv1.com",
  838: "firebaseio.com",
  839: "spamhaus.org",
  840: "deloitte.com",
  841: "primevideo.com",
  842: "hindustantimes.com",
  843: "zoho.com",
  844: "blueapron.com",
  845: "dnsv.jp",
  846: "arxiv.org",
  847: "huffpost.com",
  848: "lefigaro.fr",
  849: "academia.edu",
  850: "alidns.com",
  851: "unpkg.com",
  852: "flashtalking.com",
  853: "rspamd.com",
  854: "bitdefender.com",
  855: "nic.do",
  856: "netgear.com",
  857: "va.gov",
  858: "bild.de",
  859: "gdemoideti.ru",
  860: "disneyplus.com",
  861: "epa.gov",
  862: "azure-devices.net",
  863: "fbsbx.com",
  864: "mckinsey.com",
  865: "xfinity.com",
  866: "google.com.tr",
  867: "pornhub.com",
  868: "google.com.mx",
  869: "spotifycdn.com",
  870: "aniview.com",
  871: "markmonitor.com",
  872: "genius.com",
  873: "mdpi.com",
  874: "bluehost.com",
  875: "otto.de",
  876: "amazon.es",
  877: "arcgis.com",
  878: "apnews.com",
  879: "ebay.co.uk",
  880: "tinkoff.ru",
  881: "adblockplus.org",
  882: "kickstarter.com",
  883: "bankofamerica.com",
  884: "sonobi.com",
  885: "smartthings.com",
  886: "biblegateway.com",
  887: "discogs.com",
  888: "btloader.com",
  889: "intentiq.com",
  890: "samsungnyc.com",
  891: "heytapmobi.com",
  892: "wp.pl",
  893: "mozilla.net",
  894: "classlink.com",
  895: "wbbasket.ru",
  896: "chartbeat.net",
  897: "tokopedia.com",
  898: "ad.gt",
  899: "richaudience.com",
  900: "firefox.com",
  901: "dbankcloud.ru",
  902: "instructure.com",
  903: "aiv-delivery.net",
  904: "kaspi.kz",
  905: "allrecipes.com",
  906: "chaturbate.com",
  907: "onet.pl",
  908: "daum.net",
  909: "synology.com",
  910: "adgrx.com",
  911: "samsungosp.com",
  912: "lowes.com",
  913: "bytetcdn.com",
  914: "dns.cn",
  915: "amazon.com.br",
  916: "utorrent.com",
  917: "hm.com",
  918: "gandi-ns.fr",
  919: "docker.io",
  920: "instabug.com",
  921: "ip-api.com",
  922: "repubblica.it",
  923: "redd.it",
  924: "1drv.com",
  925: "weforum.org",
  926: "live-video.net",
  927: "shopeemobile.com",
  928: "mongodb.com",
  929: "readthedocs.io",
  930: "volcfcdndvs.com",
  931: "badoo.com",
  932: "vkuseraudio.net",
  933: "atlassian.net",
  934: "youku.com",
  935: "innovid.com",
  936: "tplinkcloud.com",
  937: "character.ai",
  938: "mirtesen.ru",
  939: "samsungcloudsolution.net",
  940: "skyhigh.cloud",
  941: "dyndns.org",
  942: "si.com",
  943: "pexels.com",
  944: "chess.com",
  945: "nintendo.net",
  946: "alphonso.tv",
  947: "ubnt.com",
  948: "elmundo.es",
  949: "onesignal.com",
  950: "statcounter.com",
  951: "hackaday.com",
  952: "rapid7.com",
  953: "xnxx.com",
  954: "corriere.it",
  955: "umeng.com",
  956: "tencent.com",
  957: "pikabu.ru",
  958: "eporner.com",
  959: "ebay.de",
  960: "columbia.edu",
  961: "openstreetmap.org",
  962: "mercadolivre.com.br",
  963: "ancestry.com",
  964: "cbc.ca",
  965: "economist.com",
  966: "amazon.com.au",
  967: "nexusmods.com",
  968: "kohls.com",
  969: "bytefcdn.com",
  970: "vmware.com",
  971: "seznam.cz",
  972: "quizlet.com",
  973: "gihc.net",
  974: "agoda.com",
  975: "kontur.ru",
  976: "caixa.gov.br",
  977: "cloudinary.com",
  978: "supersonicads.com",
  979: "apple.news",
  980: "pandora.com",
  981: "xing.com",
  982: "umich.edu",
  983: "microsoftapp.net",
  984: "dreamhost.com",
  985: "microsoftpersonalcontent.com",
  986: "mercadolibre.com.ar",
  987: "macys.com",
  988: "usnews.com",
  989: "betweendigital.com",
  990: "deepl.com",
  991: "nelreports.net",
  992: "www.gov.br",
  993: "freefiremobile.com",
  994: "chartboost.com",
  995: "costco.com",
  996: "4dex.io",
  997: "hilton.com",
  998: "digitaloceanspaces.com",
  999: "beget.com",
  1000: "wikihow.com"
}

def get_website_from_sample_name(sample_name):
    # x_y.pcap, where:
    #    --- x is the number of the website in the top 1000 tranco list
    #    --- y is the number of the sample for the given website
    number = int(sample_name.split('_')[0])
    
    return website_mapping.get(number, "Website not found")

################################################################################

# Adapted from: https://github.com/dmbb/MPTAnalysis/blob/master/CovertCastAnalysis/extractFeatures.py
def extract_features(sampleFolder, outputFolder):
    arff = open(outputFolder + '_dataset.csv', 'w')
    written_header = False

    for sample in os.listdir(sampleFolder):
        f = open(sampleFolder + "/" + sample, 'rb' )
        print(sampleFolder + "/" + sample)
        pcap = dpkt.pcap.Reader(f)

        #Analyse packets transmited
        totalPackets = 0
        totalPacketsIn = 0
        totalPacketsOut = 0

        #Analyse bytes transmitted
        totalBytes = 0
        totalBytesIn = 0
        totalBytesOut = 0

        #Analyse packet sizes
        packetSizes = []
        packetSizesIn = []
        packetSizesOut = []

        #Analyse inter packet timing
        packetTimes = []
        packetTimesIn = []
        packetTimesOut = []

        #Analyse outcoming bursts
        bursts_packets = []
        burst_sizes = []
        burst_times = []
        current_burst = 0
        current_burst_start = 0
        current_burst_size = 0

        prev_ts = 0
        absTimesOut = []

        for ts, buf in pcap:
            eth = dpkt.ethernet.Ethernet(buf)
            ip_hdr = eth.data
            try:
                if (eth.type != dpkt.ethernet.ETH_TYPE_IP and eth.type != dpkt.ethernet.ETH_TYPE_IP6):
                    continue

                # Check if the packet is either TCP or UDP and involves port 443
                if ip_hdr.p in [6, 17] and (ip_hdr.data.sport == 443 or ip_hdr.data.dport == 443):
                    # General packet statistics
                    totalPackets += 1

                    # If source port is 443, it's an incoming packet (from the server to your system)
                    if ip_hdr.data.sport == 443:
                        totalPacketsIn += 1
                        packetSizesIn.append(len(buf))

                        if prev_ts != 0:
                            ts_difference = ts - prev_ts
                            packetTimesIn.append(ts_difference)

                        # Burst logic for incoming packets
                        if current_burst != 0:
                            if current_burst > 1:
                                bursts_packets.append(current_burst)
                                burst_sizes.append(current_burst_size)
                                burst_times.append(ts - current_burst_start)
                            current_burst = 0
                            current_burst_size = 0
                            current_burst_start = 0

                    # If destination port is 443, it's an incoming packet (from the server to your system)
                    elif ip_hdr.data.dport == 443:
                        totalPacketsOut += 1
                        absTimesOut.append(ts)
                        packetSizesOut.append(len(buf))

                        if prev_ts != 0:
                            ts_difference = ts - prev_ts
                            packetTimesOut.append(ts_difference)

                        # Burst logic for outgoing packets
                        if current_burst == 0:
                            current_burst_start = ts
                        current_burst += 1
                        current_burst_size += len(buf)

                    # Bytes transmitted statistics
                    totalBytes += len(buf)
                    if ip_hdr.data.sport == 443:
                        totalBytesIn += len(buf)
                    else:
                        totalBytesOut += len(buf)

                    # Packet Size statistics
                    packetSizes.append(len(buf))

                    # Packet Times statistics
                    if prev_ts != 0:
                        ts_difference = ts - prev_ts
                        packetTimes.append(ts_difference)

                    prev_ts = ts
            except dpkt.dpkt.NeedData:
                print(f"Skipping corrupt packet in {sample}")
                continue
            except Exception as e:
                print(f"Error processing packet in {sample}: {e}")
                continue

        f.close()


        ########################################################################
        # Compute BytesPerSecond & PacketsPerSecond Timeseries
        ########################################################################

        last_index = 0
        BytesPerSecond = []
        AvgBytesPerSecond = []
        PacketsPerSecond = []
        
        if len(absTimesOut) > 0:
            t = absTimesOut[0] + 0.25
            max_time = max(absTimesOut) + 1 #! Adjust conforming to cap len.
            
            while t < max_time:
                indices = [i for i,v in enumerate(absTimesOut) if v <= t]

                bps = 0
                for i in indices[last_index:]:
                    bps += packetSizesOut[i]

                BytesPerSecond.append(bps)
                if(len(indices[last_index:]) > 0):
                    AvgBytesPerSecond.append(bps/len(indices[last_index:]))
                else:
                    AvgBytesPerSecond.append(0)
                PacketsPerSecond.append(len(indices[last_index:]))
                last_index = len(indices)
                t += 0.25

        #Write sample features to the csv file
        f_names = []
        f_values = []

        f_names.append('website')
        f_values.append(get_website_from_sample_name(sample))

        ########################################################################
        #Global Packet Features
        f_names.append('totalPackets')
        f_values.append(totalPackets)
        f_names.append('totalPacketsIn')
        f_values.append(totalPacketsIn)
        f_names.append('totalPacketsOut')
        f_values.append(totalPacketsOut)
        f_names.append('totalBytes')
        f_values.append(totalBytes)
        f_names.append('totalBytesIn')
        f_values.append(totalBytesIn)
        f_names.append('totalBytesOut')
        f_values.append(totalBytesOut)

        ########################################################################
        #Packet Length Features
        meanPacketSizes, stdevPacketSizes, variancePacketSizes, maxPacketSizes, minPacketSizes, kurtosisPacketSizes, skewPacketSizes = safe_stats(packetSizes)

        f_names.extend(['minPacketSize', 'maxPacketSize', 'meanPacketSizes', 'stdevPacketSizes', 'variancePacketSizes', 
                        'skewPacketSizes', 'kurtosisPacketSizes', 
                        'p10PacketSizes', 'p20PacketSizes', 'p30PacketSizes', 'p40PacketSizes', 'p50PacketSizes', 
                        'p60PacketSizes', 'p70PacketSizes', 'p80PacketSizes', 'p90PacketSizes'])

        f_values.extend([minPacketSizes, maxPacketSizes, meanPacketSizes, stdevPacketSizes, variancePacketSizes, 
                        skewPacketSizes, kurtosisPacketSizes,
                        safe_percentile(packetSizes, 10), safe_percentile(packetSizes, 20), safe_percentile(packetSizes, 30),
                        safe_percentile(packetSizes, 40), safe_percentile(packetSizes, 50), safe_percentile(packetSizes, 60),
                        safe_percentile(packetSizes, 70), safe_percentile(packetSizes, 80), safe_percentile(packetSizes, 90)])


        ########################################################################
        #Packet Length Features (in)
        meanPacketSizesIn, stdevPacketSizesIn, variancePacketSizesIn, maxPacketSizesIn, minPacketSizesIn, kurtosisPacketSizesIn, skewPacketSizesIn = safe_stats(packetSizesIn)

        f_names.extend(['minPacketSizeIn', 'maxPacketSizeIn', 'meanPacketSizesIn', 'stdevPacketSizesIn', 'variancePacketSizesIn', 
                        'skewPacketSizesIn', 'kurtosisPacketSizesIn', 
                        'p10PacketSizesIn', 'p20PacketSizesIn', 'p30PacketSizesIn', 'p40PacketSizesIn', 'p50PacketSizesIn', 
                        'p60PacketSizesIn', 'p70PacketSizesIn', 'p80PacketSizesIn', 'p90PacketSizesIn'])

        f_values.extend([minPacketSizesIn, maxPacketSizesIn, meanPacketSizesIn, stdevPacketSizesIn, variancePacketSizesIn, 
                        skewPacketSizesIn, kurtosisPacketSizesIn,
                        safe_percentile(packetSizesIn, 10), safe_percentile(packetSizesIn, 20), safe_percentile(packetSizesIn, 30),
                        safe_percentile(packetSizesIn, 40), safe_percentile(packetSizesIn, 50), safe_percentile(packetSizesIn, 60),
                        safe_percentile(packetSizesIn, 70), safe_percentile(packetSizesIn, 80), safe_percentile(packetSizesIn, 90)])


        ########################################################################
        #Packet Length Features (out)

        meanPacketSizesOut, stdevPacketSizesOut, variancePacketSizesOut, maxPacketSizesOut, minPacketSizesOut, kurtosisPacketSizesOut, skewPacketSizesOut = safe_stats(packetSizesOut)

        f_names.extend(['minPacketSizesOut', 'maxPacketSizesOut', 'meanPacketSizesOut', 'stdevPacketSizesOut', 'variancePacketSizesOut', 
                        'skewPacketSizesOut', 'kurtosisPacketSizesOut', 
                        'p10PacketSizesOut', 'p20PacketSizesOut', 'p30PacketSizesOut', 'p40PacketSizesOut', 'p50PacketSizesOut', 
                        'p60PacketSizesOut', 'p70PacketSizesOut', 'p80PacketSizesOut', 'p90PacketSizesOut'])

        f_values.extend([minPacketSizesOut, maxPacketSizesOut, meanPacketSizesOut, stdevPacketSizesOut, variancePacketSizesOut, 
                        skewPacketSizesOut, kurtosisPacketSizesOut,
                        safe_percentile(packetSizesOut, 10), safe_percentile(packetSizesOut, 20), safe_percentile(packetSizesOut, 30),
                        safe_percentile(packetSizesOut, 40), safe_percentile(packetSizesOut, 50), safe_percentile(packetSizesOut, 60),
                        safe_percentile(packetSizesOut, 70), safe_percentile(packetSizesOut, 80), safe_percentile(packetSizesOut, 90)])


        ########################################################################
        #Packet Timing Features

        meanPacketTimes, stdevPacketTimes, variancePacketTimes, maxIPT, minIPT, kurtosisPacketTimes, skewPacketTimes = safe_stats(packetTimes)

        f_names.extend(['minIPT', 'maxIPT', 'meanPacketTimes', 'stdevPacketTimes', 'variancePacketTimes', 
                        'skewPacketTimes', 'kurtosisPacketTimes', 
                        'p10PacketTimes', 'p20PacketTimes', 'p30PacketTimes', 'p40PacketTimes', 'p50PacketTimes', 
                        'p60PacketTimes', 'p70PacketTimes', 'p80PacketTimes', 'p90PacketTimes'])

        f_values.extend([minIPT, maxIPT, meanPacketTimes, stdevPacketTimes, variancePacketTimes, 
                        skewPacketTimes, kurtosisPacketTimes,
                        safe_percentile(packetTimes, 10), safe_percentile(packetTimes, 20), safe_percentile(packetTimes, 30),
                        safe_percentile(packetTimes, 40), safe_percentile(packetTimes, 50), safe_percentile(packetTimes, 60),
                        safe_percentile(packetTimes, 70), safe_percentile(packetTimes, 80), safe_percentile(packetTimes, 90)])


        ########################################################################
        #Packet Timing Features (in)

        meanPacketTimesIn, stdevPacketTimesIn, variancePacketTimesIn, maxPacketTimesIn, minPacketTimesIn, kurtosisPacketTimesIn, skewPacketTimesIn = safe_stats(packetTimesIn)

        f_names.extend(['minPacketTimesIn', 'maxPacketTimesIn', 'meanPacketTimesIn', 'stdevPacketTimesIn', 'variancePacketTimesIn', 
                        'skewPacketTimesIn', 'kurtosisPacketTimesIn', 
                        'p10PacketTimesIn', 'p20PacketTimesIn', 'p30PacketTimesIn', 'p40PacketTimesIn', 'p50PacketTimesIn', 
                        'p60PacketTimesIn', 'p70PacketTimesIn', 'p80PacketTimesIn', 'p90PacketTimesIn'])
        f_values.extend([minPacketTimesIn, maxPacketTimesIn, meanPacketTimesIn, stdevPacketTimesIn, variancePacketTimesIn, 
                        skewPacketTimesIn, kurtosisPacketTimesIn,
                         safe_percentile(packetTimesIn, 10), safe_percentile(packetTimesIn, 20), safe_percentile(packetTimesIn, 30),
                         safe_percentile(packetTimesIn, 40), safe_percentile(packetTimesIn, 50), safe_percentile(packetTimesIn, 60),
                         safe_percentile(packetTimesIn, 70), safe_percentile(packetTimesIn, 80), safe_percentile(packetTimesIn, 90)])
        

        ########################################################################
        #Packet Timing Features (out)
        
        meanPacketTimesOut, stdevPacketTimesOut, variancePacketTimesOut, maxPacketTimesOut, minPacketTimesOut, kurtosisPacketTimesOut, skewPacketTimesOut = safe_stats(packetTimesOut)

        f_names.extend(['minPacketTimesOut', 'maxPacketTimesOut', 'meanPacketTimesOut', 'stdevPacketTimesOut', 'variancePacketTimesOut', 
                        'skewPacketTimesOut', 'kurtosisPacketTimesOut', 
                        'p10PacketTimesOut', 'p20PacketTimesOut', 'p30PacketTimesOut', 'p40PacketTimesOut', 'p50PacketTimesOut', 
                        'p60PacketTimesOut', 'p70PacketTimesOut', 'p80PacketTimesOut', 'p90PacketTimesOut'])
        f_values.extend([minPacketTimesOut, maxPacketTimesOut, meanPacketTimesOut, stdevPacketTimesOut, variancePacketTimesOut, 
                        skewPacketTimesOut, kurtosisPacketTimesOut,
                         safe_percentile(packetTimesOut, 10), safe_percentile(packetTimesOut, 20), safe_percentile(packetTimesOut, 30),
                         safe_percentile(packetTimesOut, 40), safe_percentile(packetTimesOut, 50), safe_percentile(packetTimesOut, 60),
                         safe_percentile(packetTimesOut, 70), safe_percentile(packetTimesOut, 80), safe_percentile(packetTimesOut, 90)])
        

        ########################################################################
        #Packet number of Bursts features

        meanBurst, stdevBurst, varianceBurst, maxBurst, _, kurtosisBurst, skewBurst = safe_stats(bursts_packets)

        f_names.extend(['totalBursts', 'maxBurst', 'meanBurst', 'stdevBurst', 'varianceBurst', 'kurtosisBurst', 'skewBurst',
                        'p10Burst', 'p20Burst', 'p30Burst', 'p40Burst', 'p50Burst', 'p60Burst', 'p70Burst', 'p80Burst', 'p90Burst'])
        f_values.extend([len(bursts_packets), maxBurst, meanBurst, stdevBurst, varianceBurst, kurtosisBurst, skewBurst,
                         safe_percentile(bursts_packets, 10), safe_percentile(bursts_packets, 20), safe_percentile(bursts_packets, 30),
                         safe_percentile(bursts_packets, 40), safe_percentile(bursts_packets, 50), safe_percentile(bursts_packets, 60),
                         safe_percentile(bursts_packets, 70), safe_percentile(bursts_packets, 80), safe_percentile(bursts_packets, 90)])


        ########################################################################
        #Packet Bursts data size features

        meanBurstBytes, stdevBurstBytes, varianceBurstBytes, maxBurstBytes, minBurstBytes, kurtosisBurstBytes, skewBurstBytes = safe_stats(burst_sizes)

        f_names.extend(['maxBurstBytes', 'minBurstBytes', 'meanBurstBytes', 'medianBurstBytes', 'stdevBurstBytes', 
                        'varianceBurstBytes', 'kurtosisBurstBytes', 'skewBurstBytes', 
                        'p10BurstBytes', 'p20BurstBytes', 'p30BurstBytes', 'p40BurstBytes', 'p50BurstBytes', 
                        'p60BurstBytes', 'p70BurstBytes', 'p80BurstBytes', 'p90BurstBytes'])
        f_values.extend([maxBurstBytes, minBurstBytes, meanBurstBytes, np.median(burst_sizes), stdevBurstBytes, 
                         varianceBurstBytes, kurtosisBurstBytes, skewBurstBytes,
                         safe_percentile(burst_sizes, 10), safe_percentile(burst_sizes, 20), safe_percentile(burst_sizes, 30),
                         safe_percentile(burst_sizes, 40), safe_percentile(burst_sizes, 50), safe_percentile(burst_sizes, 60),
                         safe_percentile(burst_sizes, 70), safe_percentile(burst_sizes, 80), safe_percentile(burst_sizes, 90)])


        if(not written_header):
            arff.write(','.join(f_names))
            arff.write('\n')
            written_header = True

        l = []
        for v in f_values:
            l.append(str(v))
        arff.write(','.join(l))
        arff.write('\n')
    arff.close()


################################################################################

if __name__ == "__main__":
    sampleFolders = [
        DATASET_FOLDER
    ]

    for sampleFolder in sampleFolders:
        print("\n#############################")
        print ("Parsing " + sampleFolder)
        print ("#############################")
        extract_features(sampleFolder, FEATURES_RESULT_PATH)
        
        shutil.rmtree(sampleFolder)
        print(f"Deleted PCAP dataset folder: {sampleFolder}")