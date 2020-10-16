// 폰트 로드
var link = document.createElement( "link" );
link.href = "https://fonts.googleapis.com/css2?family=Noto+Sans+KR:wght@300;500&display=swap";
link.rel = "stylesheet";
document.getElementsByTagName( "head" )[0].appendChild( link );

// 특수 키코드 (ex: Alt, Ctrl, F1, ESC 등등)
specialKeyCodes = [27, 9, 20, 16, 17, 91, 18, 21, 93, 25, 112, 113, 114, 115, 116, 117, 118, 119, 120, 121, 122, 123, 44, 145, 19, 45, 36, 33, 34, 35, 37, 38, 39, 40, 144];

currentInformEle = null;  // 현재 팝업이 가르키고 있는 element
async function setInform(ele) {
    // 수정 제안하는 팝업 띄우기
    var to = ele.getAttribute('to');
    currentInformEle = ele;
    $('.curse-inform-container-head').text('아래처럼 수정해 보세요:');
    if (to == '') {
        // 해당 단어를 지우라고 권고하는 경우 -> 취소선 긋기
        $('.curse-inform-container-body').css('text-decoration', 'line-through');
        $('.curse-inform-container-body').text(ele.textContent);
    } else {
        $('.curse-inform-container-body').css('text-decoration', 'none');
        $('.curse-inform-container-body').text(to);
    }
}

function sleepMS(ms) {
    // ms 만큼 sleep
    return new Promise(resolve => setTimeout(resolve, ms));
}

for (var i in $('textarea').toArray()) {
    // textarea에 hightlights 섹션 추가
    // hightlight는 단어에 밑줄을 긋는 역할을 수행한다.
    var _element = $('textarea')[i];
    var parent = _element.parentNode;
    var container = document.createElement('div');
    container.className = "container";
    parent.replaceChild(container, _element);
    container.appendChild(_element);
    var backdrop = document.createElement('div');
    backdrop.className = "backdrop";
    var highlights = document.createElement('div');
    highlights.className = "highlights";
    backdrop.appendChild(highlights);
    container.appendChild(backdrop);
    
    var curseInformContainer = document.createElement('div');
    curseInformContainer.className = "curse-inform-container";
    var curseInformContainerHead = document.createElement('div');
    curseInformContainerHead.className = "curse-inform-container-head";
    var curseInformContainerBody = document.createElement('div');
    curseInformContainerBody.className = "curse-inform-container-body";
    curseInformContainer.appendChild(curseInformContainerHead);
    curseInformContainer.appendChild(curseInformContainerBody);
    document.getElementsByTagName('body')[0].appendChild(curseInformContainer);
}

var $container = $('.container');
var $backdrop = $('.backdrop');
var $highlights = $('.highlights');
var $textarea = $('textarea');

$('*').css('box-sizing', 'border-box');
$('.backdrop').css('position', 'absolute');
$('.backdrop').css('z-index', '1');
$('.highlights').css('white-space', 'pre-wrap');
$('.highlights').css('word-wrap', 'break-word');
$('.highlights').css('color', 'transparent');
$('textarea').css('display', 'block');
$('textarea').css('position', 'absolute');
$('textarea').css('z-index', '0');

$('.curse-inform-container').css('display', 'none');
$('.curse-inform-container').css('width', '330px');
$('.curse-inform-container').css('position', 'fixed');
$('.curse-inform-container').css('padding', '15px');
$('.curse-inform-container').css('padding-bottom', '20px');
$('.curse-inform-container').css('background', 'white');
$('.curse-inform-container').css('z-index', '3');
$('.curse-inform-container').css('border-radius', '7px');
$('.curse-inform-container').css('box-shadow', '0 0 12px rgba(220, 220, 220, 0.7)');
$('.curse-inform-container').css('flex-direction', 'column');
$('.curse-inform-container').css('opacity', '1');
$('.curse-inform-container').css('cursor', 'pointer');

$('.curse-inform-container-head').css('margin-bottom', '4px');
$('.curse-inform-container-head').css('font-family', 'Noto Sans KR');
$('.curse-inform-container-head').css('font-size', '16px');
$('.curse-inform-container-head').css('font-weight', '300');
$('.curse-inform-container-head').css('color', 'rgb(150, 150, 150)');

$('.curse-inform-container-body').css('font-family', 'Noto Sans KR');
$('.curse-inform-container-body').css('font-size', '22px');
$('.curse-inform-container-body').css('font-weight', '500');
$('.curse-inform-container-body').css('color', 'rgb(240, 100, 100)');
$('.curse-inform-container-body').css('text-decoration-color', 'gray');

function markCSS() {
    $('mark').css('color', 'transparent');
    $('mark').css('background-color', 'rgba(252, 220, 220, 0)');
    $('mark').css('border-bottom', '2px solid rgb(240, 100, 100)');
    $('mark').css('position', 'relative');
    for (var i = 0; i < document.getElementsByTagName('mark').length; i++) {
        var ele = document.getElementsByTagName('mark')[i];
        ele.animate([{borderBottomColor:'transparent'}, {borderBottomColor:'rgb(240, 100, 100)'}], 300);
    }
}

$('.container, .backdrop').css('width', '100%');
$('.container, .backdrop, textarea').css('height', '102px'); 

$('.highlights, textarea').css('padding', '9px 10px'); 
$('.highlights, textarea').css('font', '16px "Apple SD Gothic Neo", "Malgun Gothic", "맑은 고딕", Dotum, "돋움", "Noto Sans KR", "Nanum Gothic", "Lato", Helvetica, sans-serif'); 
$('.highlights, textarea').css('letter-spacing', 'normal');

$('.backdrop').css('margin', $('textarea').css('margin'));
$('.backdrop').css('color', $('textarea').css('color'));
$('.backdrop').css('resize', $('textarea').css('resize'));
$('.backdrop').css('border', $('textarea').css('border'));
$('.backdrop').css('background-color', 'transparent');
$('.backdrop').css('overflow', 'visible');  // 이렇게 설정하면 사각형이 잘리지 않는데 대신 스크롤 할 경우 이상 발생함!
$('.backdrop').css('pointer-events', 'none');



function handleInput() {
    // 모든 마크 초기화
    $highlights.html('');
}

function handleScroll() {
    var scrollTop = $textarea.scrollTop();
    $backdrop.scrollTop(scrollTop);
    
    var scrollLeft = $textarea.scrollLeft();
    $backdrop.scrollLeft(scrollLeft);  
}

function bindEvents() {
     $textarea.on({
         'input': handleInput,
         'scroll': handleScroll
     });
}

bindEvents();
handleInput();

animatinG = false;
hash = null;

$('textarea').on('keyup', async function(e) {
    if (specialKeyCodes.includes(e.keyCode)) {
        // 텍스트를 입력한 게 아니라면? -> 종료
        return 0;
    }
    let curHash = Math.random() * 100000;  // 고유 코드 (다른 listener가 실행됐는지 확인하기 위함)
    hash = curHash;
    await sleepMS(500);
    if (curHash === hash) {
        $.ajax({
            url: 'http://localhost:8000/polish',
            type: 'POST',
            data: {'text': e.currentTarget.value},
            success: onSuccess(e)
        });
    }
});

$('.curse-inform-container').on('click', (ev) => {
    var to = currentInformEle.getAttribute('to');
    var children = currentInformEle.parentElement.children;
    var nextEle = undefined;
    var value = '';
    var just = false;
    for (var i in children) {
        if (children[i] == currentInformEle) {
            var nextEle = children[parseInt(i)+1];
            value += to;
            just = true;
        } else if (children[i].textContent !== undefined) {
            var text = children[i].textContent
            if (just) {
                if (text[0] === ' ' && to === '') {
                    // 단어를 삭제할 경우 띄어쓰기가 두 개가 되므로 하나로 단축
                    text = text.slice(1, text.length)
                }
                just = false;
            }
            value += text;
        }
    }

    if (to === '' && nextEle) {
        let innerhtml = nextEle.innerHTML;
        if (innerhtml[0] === ' ') {
            // 띄어쓰기 두 개 되는 문제를 해결했을 때 hightlight 섹션도 똑같이 적용
            nextEle.innerHTML = innerhtml.slice(1, innerhtml.length);
        }
    }
    // 수정 내용 적용
    currentInformEle.outerHTML = "<plain>" + to + "</plain>";
    $('textarea').val(value);

    $('.curse-inform-container').css('opacity', '0');
    $('.curse-inform-container')[0].animate([{opacity:1}, {opacity:0}], 100);
    setTimeout(function() {$('.curse-inform-container').css('display', 'none');}, 100);
})


function onSuccess(e) {
    console.log(e.currentTarget);
    var currentText = e.currentTarget.value;
    
    return (data) => {
        console.log(data);
        if (data.result !== currentText) {
            // 수정됐을 때
            console.log('currentText', currentText);
            var html = "<plain>" + data.pointed;
            for (var i in data['result']) {
                console.log(i);
                html = html.replace('[P' + i + ']', "</plain><mark style='position:relative' class='curse-curse' to='" + data['result'][i] + "'>" + data['org'][i] +"</mark><plain>");
            html += "</plain>";
            $highlights.html(html);
            markCSS();

            e.currentTarget.addEventListener('mousemove', (ev) => {
                var nothing = true; // 아무것도 닿지 않았나?
                for (var i = 0; i < $('mark').length; i++) {
                    var ele = $('mark')[i];
                    var x = ev.clientX;
                    var y = ev.clientY;
                    var rect = ele.getBoundingClientRect();
                    if (rect.left <= x && x <= rect.right && rect.top <= y && y <= rect.bottom) {
                        if (animatinG - 10 !== i && $($('.curse-curse')[i]).css('background-color') === 'rgba(252, 220, 220, 0)') {
                            // 마우스가 특정 mark에 닿았을 때 (=ele에 닿았을 때)
                            animatinG = i + 10;
                            $($('.curse-curse')[i]).css('background-color', 'rgba(252, 220, 220, 1)');
                            $('.curse-curse')[i].animate([{backgroundColor:'rgba(252, 220, 220, 0)'}, {backgroundColor:'rgba(252, 220, 220, 1)'}], 100);
                            $($('.curse-curse')[i]).css('color', $('textarea').css('color'));
                            setTimeout(function() {animatinG = 0}, 100);
                        }
                    } else if ($($('.curse-curse')[i]).css('background-color') === 'rgb(252, 220, 220)' || animatinG-10 === i) {
                        animatinG = i + 10;
                        $($('.curse-curse')[i]).css('background-color', 'rgba(252, 220, 220, 0)');
                        $('.curse-curse')[i].animate([{backgroundColor:'rgba(252, 220, 220, 1)'}, {backgroundColor:'rgba(252, 220, 220, 0)'}], 100);
                        $($('.curse-curse')[i]).css('color', 'transparent');
                        $('.curse-curse')[i].animate([{color:'rgba(0, 0, 0, 1)'}, {backgroundColor:'rgba(0, 0, 0, 0)'}], 100);
                        setTimeout(function() {animatinG = 0}, 100);
                    }
                }
            });

            e.currentTarget.addEventListener('click', (ev) => {
                var nothing = true; // 아무것도 닿지 않았나?
                for (var i = 0; i < $('mark').length; i++) {
                    var ele = $('mark')[i];
                    var x = ev.clientX;
                    var y = ev.clientY;
                    var rect = ele.getBoundingClientRect();
                    if (rect.left <= x && x <= rect.right && rect.top <= y && y <= rect.bottom) {
                        // 마우스가 특정 mark에 닿았을 때 (=ele에 닿았을 때)
                        $('.curse-inform-container').css('left', x);
                        $('.curse-inform-container').css('top', rect.bottom + 8);
                        $('.curse-inform-container').css('display', 'block');
                        $('.curse-inform-container').css('opacity', '1');
                        setInform(ele);
                        nothing = false;
                    }
                }
                if (nothing) {
                    
                    if ($('.curse-inform-container').css('opacity') == '1') {
                        // 직전까지는 정보창 나오고 있었으면 -> 애니메이션으로 사라지기
                        $('.curse-inform-container').css('opacity', '0');
                        $('.curse-inform-container')[0].animate([{opacity:1}, {opacity:0}], 100);
                        setTimeout(function() {$('.curse-inform-container').css('display', 'none');}, 100);
                    }
                }
            })

            
            }
        }
    }
}
