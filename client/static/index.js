$("#signup").click(function() {
    $("#first").fadeOut("fast", function() {
    $("#second").fadeIn("fast");
    });
    });
    
$("#signin").click(function() {
$("#second").fadeOut("fast", function() {
$("#first").fadeIn("fast");
});
});
    
    
      
$(function() {
  $("form[name='login']").validate({
    rules: {
      
      loginusername: {
        required: true,
      },
      loginpassword: {
        required: true,
        
      }
    },
    messages: {
    
      loginpassword: {
        required: "Please enter password",
      
      }
      
    },
    submitHandler: function(form,e) {
      e.preventDefault()
      console.log(form.loginusername.value);
      localStorage.setItem("username",form.loginusername.value)
      window.location.href="/login"
    }
  });
});
             
    
    
$(function() {
  
  $("form[name='registration']").validate({
    rules: {
      firstname: "required",
      lastname: "required",
      username:"required",
      email: {
        required: true,
        email: true
      },
      password: {
        required: true,
        minlength: 5
      }
    },
    
    messages: {
      firstname: "Please enter your firstname",
      lastname: "Please enter your lastname",
      password: {
        required: "Please provide a password",
        minlength: "Your password must be at least 5 characters long"
      },
      email: "Please enter a valid email address"
    },
  
    submitHandler: function(form,e) {
      e.preventDefault()
      console.log(form.username.value);
      localStorage.setItem("username",form.username.value)
      window.location.href = "/registerface"
    }
  });
});
    