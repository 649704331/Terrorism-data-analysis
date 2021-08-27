$(".list").click(function (e) {
	$(".list a").removeClass("active");
	var tar=$(e.target);
	e.preventDefault();
	tar.addClass("active");
	var id=tar.attr("href");
	$(".detail>div").addClass("hidden");
	$("#"+id).removeClass("hidden");
});