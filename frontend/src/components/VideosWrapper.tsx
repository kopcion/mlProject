import axios from 'axios';
import React, { useEffect, useState, useRef } from 'react';
import { LazyLoadComponent } from 'react-lazy-load-image-component';

const firstChoiceStyle = [
  "outline outline-offset-0 outline-green-500 shadow-2xl shadow-green-400",
  "outline outline-offset-0 outline-red-500 opacity-75 shadow-2xl shadow-red-400",
  "outline outline-offset-0 outline-green-300 shadow-xl shadow-green-300",
  "outline outline-offset-0 outline-black-500 opacity-75"
];

const secondChoiceStyle = [
  "outline outline-offset-0 outline-red-500 opacity-75 shadow-2xl shadow-red-400",
  "outline outline-offset-0 outline-green-500 shadow-2xl shadow-green-400",
  "outline outline-offset-0 outline-green-300 shadow-xl shadow-green-300",
  "outline outline-offset-0 outline-black-500 opacity-75"
];

function useForceUpdate(){
  const [value, setValue] = useState(0); // integer state
  return () => setValue(value => value + 1); // update the state to force render
}

const VideosContainer: React.FC<{idx:number}> = ({ idx }) => {
  const forceUpdate = useForceUpdate();
  const [display, setDisplay] = useState<boolean>(false); 
  const [choice, setChoice] = useState<number>(0);
  const videoOne = useRef<HTMLVideoElement>();
  const videoTwo = useRef<HTMLVideoElement>();
  useEffect(()=>{
    axios.get(`http://127.0.0.1:8080/video_one_${idx}.mp4`)
      .then(() => {
        setDisplay(true);
        forceUpdate();
      })
      .catch(()=>{
        setDisplay(false);
        forceUpdate();
      });
  },[]);
  useEffect(()=>{
    axios.get(`/human_choice/${idx}`)
      .then((res) => {
        setChoice(res.data.choice-1);
        forceUpdate();
      })
      .catch(()=>{
        setDisplay(false);
        forceUpdate();
      });
  },[]);
  
  return (
    <>
      {display ?
        <LazyLoadComponent threshold={20}>
          <div className="flex flex-row my-4">
            <div className="flex flex-col">
              <div className="flex flex-row mx-auto pb-2">
                Episode {idx}
              </div>
              <div className="flex flex-row">
                <video className={`rounded-xl mx-4 ${firstChoiceStyle[choice]}`} autoPlay loop muted width="140" height="140" src={`http://127.0.0.1:8080/video_one_${idx}.mp4`}/>
                <video className={`rounded-xl mx-4 ${secondChoiceStyle[choice]}`} autoPlay loop muted width="140" height="140" src={`http://127.0.0.1:8080/video_two_${idx}.mp4`}/>
              </div>
            </div>
          </div>
        </LazyLoadComponent>
        :
        <></>
      }
    </>
  );
};

export default VideosContainer;
