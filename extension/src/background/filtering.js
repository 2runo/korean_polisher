
async function filtering(tab) {
    chrome.tabs.executeScript(tab.id, {code: window['jquery']});  // 탭에서 jquery 실행
    chrome.tabs.executeScript(tab.id, {code: window['tracker']});  // 탭에서 /src/website/tracker.js 실행
}
