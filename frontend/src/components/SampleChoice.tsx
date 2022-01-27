import React, { useEffect, useState, useRef } from 'react';
import axios from 'axios';
import axiosRetry from 'axios-retry';

function useForceUpdate(){
  const [value, setValue] = useState(0); // integer state
  return () => setValue(value => value + 1); // update the state to force render
}
const sleep = (milliseconds:number) => {
  return new Promise(resolve => setTimeout(resolve, milliseconds))
}

const SampleChoice: React.FC = () => {
  const forceUpdate = useForceUpdate();
  const nextEpisodeAvailable = useRef<boolean>(false);
  const [loading, setLoading] = useState<boolean>(false);
  const episodeNumber = useRef<number>(0);
  const submitChoice = (choice: number) => {
    axios.post(`/human_choice/${choice}/episode/${episodeNumber.current}`)
      .then(() => {
        nextEpisodeAvailable.current = false;
        forceUpdate();
      });
  };

  useEffect(()=>{},[]);

  const checkNewEpisode = () => {
    new Promise(async ()=>{
      for (var i = 0; i < 200; i++) {
        await axios.get('/episode_number')
          .then((res) => {
            if(episodeNumber.current < res.data.episode_number){
              console.log(res.data);
              episodeNumber.current = res.data.episode_number;
              nextEpisodeAvailable.current = !res.data.voted;
              i=200;
              setLoading(false);
            }
          })
          .catch(() => {
          });
        await sleep(1000);
      }
      setLoading(false);
    });
    setLoading(true);
  };
  const buttonStyle = "bg-indigo-500 hover:bg-blue-500 p-2 active:bg-cyan-500 rounded-xl";
  return (
    <>
      {nextEpisodeAvailable.current ? 
        <>
          <div className="flex flex-col mx-auto mt-28">
            <div className="flex flex-row mx-auto mb-10">
                Episode {episodeNumber.current}
            </div>
            <div className="flex flex-row">
              <div className="flex flex-col mx-4">
                <video onLoad={()=>(document.querySelectorAll("video").forEach(v=>(v.playbackRate=0.7)))} className="rounded-xl" autoPlay loop muted width="320" height="320" src={`http://127.0.0.1:8080/video_one_${episodeNumber.current}.mp4`}/>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(1))}>1</button>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(3))}>Equally good</button>
              </div>
              <div className="flex flex-col mx-4">
                <video onLoad={()=>(document.querySelectorAll("video").forEach(v=>(v.playbackRate=0.7)))} className="rounded-xl" autoPlay loop muted width="320" height="320" src={`http://127.0.0.1:8080/video_two_${episodeNumber.current}.mp4`}/>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(2))}>2</button>
                <button className={`${buttonStyle} mt-2`} onClick={()=>(submitChoice(4))}>Discard</button>
              </div>
            </div>
          </div>
        </>
      :
        <div className="flex justify-items-center w-screen h-screen">
          {loading ? 
          <button type="button" className={`${buttonStyle} m-auto flex flex-row bg-blue-500`} disabled>
            <svg className="animate-spin h-10 w-10 mr-3" viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
              <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" stroke-width="4"></circle>
              <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
            </svg>
            <p className="my-auto">Next episode is being rendered...</p>
          </button>
          : 
          <button onClick={checkNewEpisode} className={`${buttonStyle} m-auto`}>
            Go to the next episode
          </button>
          }
          </div>
        }
    </>
  );
};

export default SampleChoice;
