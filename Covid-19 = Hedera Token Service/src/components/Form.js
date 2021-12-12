import React, { useState } from 'react';
import firebase from '../util/firebase';
 
export default function Form() {
  const [patientName, setTitle] = useState('');
  const [symptoms, getSymptoms] = useState('');
  const [age, getAge] = useState('');
  const [phonenumber, getNumber] = useState('');
  const [email, getEmail] = useState('');
  const [location, getLocation] = useState('');

  
  const handleOnChange = (e) => {
    setTitle(e.target.value);
    
  };

  const handleChange=e=>{
    getSymptoms(e.target.value);
  }

  const handleAge=e=>{
    getAge(e.target.value);
  }
  const handleNumber=e=>{
    getNumber(e.target.value);
  }

  const handleEmail=e=>{
    getEmail(e.target.value);
  }

  
  const handleLocation=e=>{
    getLocation(e.target.value);
  }


  const createTodo = () => {
    const todoRef = firebase.database().ref('Covid-19 Patient Details');
    const todo = {
      patientName,
      symptoms,
      age, 
      phonenumber,
      email,
      location,
      // newAccountId,
      status: false,
    };

    todoRef.push(todo);
    alert("The Patient Details are Successfully Submitted to Hedera")
  };
  return (
  <div>
<div class="row">
      <div class="col-25">
        <label for="fname"><h5>Patient Name</h5></label>
      </div>
      <div class="col-75">
        
        <input type="text" id="pname" name="patientName" onChange={handleOnChange} value={patientName}  placeholder="Enter your name.." style={{  width:"300px"}}  />
      </div>
    </div>

    <div class="row">
      <div class="col-25">
        <label for="subject" style={{marginTop:"1rem"}}><h5>Symptoms</h5></label>
      </div>
      <div class="col-75">
        <textarea id="subject"onChange={handleChange} value={symptoms} name="Symptoms" placeholder="Write about your symptoms.." style={{height:"100px",width:"300px" }}  ></textarea>
      </div>
    </div>

    <div class="row">
      <div class="col-25">
        <label for="age"   ><h5>Age</h5></label>
      </div>
      <div class="col-75">
        
        <input type="number" id="age" name="age" onKeyDown={ (evt) => evt.key === 'e' && evt.preventDefault() }  onChange={handleAge} value={age}  placeholder="Enter your Age.." style={{  width:"300px"}}  />
      </div>
    </div>

    <div class="row">
      <div class="col-25">
        <label for="phone-number"style={{marginTop:"1rem"}}><h5>Phone Number</h5></label>
      </div>
      <div class="col-75">
        
        <input type="number" id="phone-number" onKeyDown={ (evt) => evt.key === 'e' && evt.preventDefault() } name="number" onChange={handleNumber} value={phonenumber}  placeholder="Enter your Phone Number.." style={{  width:"300px"}}  />
      </div>
    </div>

    <div class="row">
      <div class="col-25">
        <label for="email" style={{marginTop:"1rem"}}><h5>Email</h5></label>
      </div>
      <div class="col-75">
        
        <input type="email" id="email" name="email" onChange={handleEmail} value={email}  placeholder="Enter your Email.." style={{  width:"300px"}} required />
      </div>
    </div>
 

    <div class="row">
      <div class="col-25">
        <label for="location" style={{marginTop:"1rem"}}><h5>Location</h5></label>
      </div>
      <div class="col-75">
        
        <input type="url" id="location" name="location" onChange={handleLocation} value={location}  placeholder="Paste Your URL Here.." 
        style={{  width:"300px"}} required />
      </div>
    </div>
    

    <div class="row">

    <div class="col-75">
    <button className="sub"
    style={{color:"white", width:"300px",height:"40px", borderRadius:"2rem", border:"none", background:"#3366ff",marginTop:"2rem"}} 
    onClick={createTodo}>Add Patient</button>
    <h6 style={{marginTop:"1rem",fontWeight:"bolder"}}>Click here to submit!</h6>
    </div>

    </div>
  </div>
  );
}
