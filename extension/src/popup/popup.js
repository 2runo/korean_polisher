var toggle = document.getElementsByClassName("onoffToggle");
for (var i=0; i<toggle.length; i++) {
    toggle[i].addEventListener("mousedown", function() {
        if (this.className == "onoffToggle on") {
            this.className = "onoffToggle off";
            this.getElementsByTagName("span")[0].textContent = "OFF";
            this.parentNode.getElementsByTagName('span')[0].style.opacity = "0";
            document.getElementsByTagName("article")[0].className = "ffo";
            var value = "off";
        } else if (this.className == "onoffToggle off") {
            this.className = "onoffToggle on";
            this.getElementsByTagName("span")[0].textContent = "ON";
            this.parentNode.getElementsByTagName('span')[0].style.opacity = "1";
            document.getElementsByTagName("article")[0].className = "no";
            var value = "on";
        }
        chrome.storage.sync.set({turn: value});  //  설정 저장
})
}


function Start() {
    // 설정에서 Korean Polisher 켜져 있나 꺼져 있나 확인하여 자동으로 켜거나 끄기
    chrome.storage.sync.get('turn', function(data) {
        if (Object.keys(data).length == 0) {
            var value = 'on';
            chrome.storage.sync.set({ turn: value });
        } else {
            var value = data['turn'];
        }
        if (value == "on") {
            onStart();
        } else if (value == "off") {
            offStart();
        }
    });
}
function onStart() {
    // Korean Polisher 켜기
    document.getElementsByTagName('article')[0].className = "no";
    document.getElementById("onoffCont").getElementsByClassName("onoffToggle")[0].className = "onoffToggle on";
}
function offStart() {
    // Korean Polisher 끄기
    document.getElementsByTagName('article')[0].className = "ffo";
    document.getElementById("onoffCont").getElementsByClassName("onoffToggle")[0].className = "onoffToggle off";
}

Start();
