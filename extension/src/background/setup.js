// 소스 가져오기

var xhr = new XMLHttpRequest();
xhr.open('GET', chrome.extension.getURL('/src/background/jquery.min.js'), true);
xhr.onreadystatechange = function()
{
    if(xhr.readyState == XMLHttpRequest.DONE && xhr.status == 200)
    {
        window['jquery'] = xhr.responseText;
        // jquery code
        var xhr2 = new XMLHttpRequest();
        xhr2.open('GET', chrome.extension.getURL('/src/website/tracker.js'), true);
        xhr2.onreadystatechange = function()
        {
            if(xhr2.readyState == XMLHttpRequest.DONE && xhr2.status == 200)
            {
                window['tracker'] = xhr2.responseText;
                // jquery code
                var xhr3 = new XMLHttpRequest();
                xhr3.open('GET', chrome.extension.getURL('/src/background/jquery-ui.min.js'), true);
                xhr3.onreadystatechange = function()
                {
                    if(xhr3.readyState == XMLHttpRequest.DONE && xhr3.status == 200)
                    {
                        window['jquery-ui'] = xhr3.responseText;
                    }
                };
                xhr3.send();
            }
        };
        xhr2.send();
    }
};
xhr.send();
