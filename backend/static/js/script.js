function encodeImageFileAsURL(element) {
    var file = element.files[0];  
    var reader = new FileReader(); 
    var loader = "/static/images/loader.gif";
    reader.onloadend = function() {  
        var imagebase64 = reader.result;  
        $("#original").attr('src', imagebase64);
        $("#age-detect").attr('src', loader);
        $("#colourise").attr('src', loader);
        $.post("http://localhost:8000/color/", JSON.stringify({"img": imagebase64}), function(response){
            var colorB64 = "data:image/jpeg;base64," + response.img;
            $("#colourise").attr('src', colorB64);
            $.post("http://localhost:8000/age/", JSON.stringify({"img": colorB64}), function(response){
                var ageB64 = "data:image/jpeg;base64," + response.img;
                $("#age-detect").attr('src', ageB64);
            })
        })
        
    }  
    reader.readAsDataURL(file);
}

function b64_to_bin(str) {
    console.log(str);
    var binstr = atob(str)
    var bin = new Uint8Array(binstr.length)
    for (var i = 0; i < binstr.length; i++) {
        bin[i] = binstr.charCodeAt(i)
    }
    return bin
}

function downloadColourised(){
    var data_b64 = $("#colourise").attr('src').replace(/^data:image\/jpeg;base64,/, "");
    var data = b64_to_bin(data_b64);
    var blob = new Blob([data], {type: "application/octet-stream"})
    var url = window.URL.createObjectURL(blob)
    var a = document.createElement("a")
    a.href = url
    a.download = "colourise.png"
    var event = document.createEvent("MouseEvents");
    event.initEvent("click", true, true);
    a.dispatchEvent(event);
}

function downloadAgeDetected(){
    var data_b64 = $("#age-detect").attr('src').replace(/^data:image\/jpeg;base64,/, "");
    var data = b64_to_bin(data_b64);
    var blob = new Blob([data], {type: "application/octet-stream"})
    var url = window.URL.createObjectURL(blob)
    var a = document.createElement("a")
    a.href = url
    a.download = "age-detection.png"
    var event = document.createEvent("MouseEvents");
    event.initEvent("click", true, true);
    a.dispatchEvent(event);
}

$(document).ready(function(){
    $("#original").attr('src', base64);
    var fileupload = $("#fileUpload");
    var button = $("#btnFileUpload");
    button.click(function () {
        fileupload.click();
    });

    $.post("http://localhost:8000/color/", JSON.stringify({"img": base64}), function(response){
        var colorB64 = "data:image/jpeg;base64," + response.img;
        $("#colourise").attr('src', colorB64);
        $.post("http://localhost:8000/age/", JSON.stringify({"img": colorB64}), function(response){
            var ageB64 = "data:image/jpeg;base64," + response.img;
            $("#age-detect").attr('src', ageB64);
        })
    })
    
});