

function sleep(ms) {
    return new Promise(resolve => setTimeout(resolve, ms));
}

function send(msg) {
    stuck = msg;
    chrome.runtime.sendMessage(msg, function(response) {
    });
}

ret = null;
async function getSync2Var(name, def) {
    // 'name'에 해당하는 설정값을 'ret' 변수에 담기
    // 'def'는 해당 설정값이 존재하지 않을 때 자동으로 지정할 값
    chrome.storage.sync.get(name, function(data) {
        if (Object.keys(data).length == 0) {
            // 해당 설정값이 없을 경우
            ret = def;
            var mem = {};
            mem[name] = def;
            chrome.storage.sync.set(mem);
        } else {
            ret = data[name];
        }
    });
}
async function getSync(name, def) {
    // 'name'에 해당하는 설정값을 return
    // 'def'는 해당 설정값이 존재하지 않을 때 자동으로 지정할 값
    ret = null;
    //await getTurn();
    await getSync2Var(name, def);
    while (ret == null) {
        await sleep(100);
    }
    return ret;
}
async function isTurnOn() {
    // Korean Polisher 켜져 있는가?
    var setting = await getSync('turn', 'on');
    if (setting == 'on')
        return true;
    else
        return false;
}
